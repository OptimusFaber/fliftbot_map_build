import cv2
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from utils.tracker.bot_sort import STrack, BoTSORT
from utils.fourier_decoder import FourierDecoder
from ultralytics import YOLO
from utils.tracker.tracking_utils.args import args as tracker_args
from utils.utils import save_video_data
import torch
import line_profiler
import numpy as np
from numpy import typing as npt
import yaml

AUTO_MAP_DIR = Path(__file__).resolve().parent
_NAV_ROOT = AUTO_MAP_DIR.parent / "fliftbot_navigation-navigation_INS@7014eb114e3"
if _NAV_ROOT.is_dir():
    sys.path.insert(0, str(_NAV_ROOT))
# Local checkout used by this workspace.  The previous optional path only
# exists in an older export, so without this entry CameraModel silently became
# unavailable and observations were written with ``ray: null``.
_LOCAL_NAV_ROOT = AUTO_MAP_DIR.parent / "BitBucket"
if _LOCAL_NAV_ROOT.is_dir():
    sys.path.insert(0, str(_LOCAL_NAV_ROOT))

try:
    from fliftbot_navigation.camera_model import CameraModel
except Exception:
    try:
        from camera_model import CameraModel
    except Exception:
        CameraModel = None



def _to_python_list(values, cast_type):
    if isinstance(values, torch.Tensor):
        if values.is_cuda:
            values = values.detach().cpu().numpy().tolist()
        else:
            values = values.detach().numpy().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [cast_type(v) for v in values]


class VideoProcessor:
    # Keep the same geometric detection gate as the detector-only exporter.
    # A clipped box has a biased center and therefore produces a bad camera
    # ray even when BoT-SORT manages to associate it with the correct track.
    MIN_ASPECT_RATIO = 0.72
    EDGE_EPSILON_PX = 1.0
    ROBUST_SIZE_SIGMA = 3.0

    def __init__(self, video_path, model_path, bs_args, labels):
        self.args = bs_args
        self.input_path = Path(video_path)
        self.source_dir: Path | None = self.input_path if self.input_path.is_dir() else None
        self.video_path, self.frame_timestamps = self._resolve_video_input(self.input_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.detector = YOLO(model=model_path, task="detect")
        self.tracker = BoTSORT(self.args)
        self.labels = labels
        self.fourier_decoder = (
            FourierDecoder.from_labels2coords(self.labels, top_n=-1) if self.labels else None
        )
        self.image = None
        self.current_coordinates = None
        self.current_frame_timestamp: float | None = None
        self.current_frame_camera_meta: dict | None = None
        self.frame_index = 0
        camera_search_dir = self.source_dir if self.source_dir is not None else self.input_path.parent
        self.camera_model_for_export = self._build_camera_model_from_sidecars(
            Path(bs_args.camera_json), Path(bs_args.camera_position)
        )
        self.camera_origin_for_export = self._precompute_camera_origin(self.camera_model_for_export)
        if self.camera_model_for_export is None:
            print("Camera model unavailable; observations will have ray=null")
        else:
            print(f"camera_origin: {self.camera_origin_for_export}")
        # Track previous decoded codes per track id for adjacency-based narrowing
        self.prev_codes: list[int] = []
        # Build adjacency if assignment provided
        self.code_adjacency: dict[int, set[int]] = {}
        if getattr(self.args, 'assignment', None):
            try:
                with open(self.args.assignment, 'r') as f:
                    assignment_yaml = yaml.safe_load(f)
                # Expect assignment_yaml['assignment'] list with 'code'
                self.code_adjacency = {entry['code']: set(entry['adjacent_codes']) for entry in assignment_yaml.get('assignment', []) if 'code' in entry}
            except Exception as e:
                print(f"Failed to load assignment adjacency: {e}")
                self.code_adjacency = {}

    @staticmethod
    def _resolve_video_input(input_path: Path) -> tuple[str, list[float] | None]:
        """Resolve either a plain video file path or an async-writer output folder."""
        if not input_path.is_dir():
            return str(input_path), None

        base = input_path.name
        candidates = [input_path / f"{base}.mp4", input_path / f"{base}.avi"]
        video_path = next((c for c in candidates if c.exists()), None)
        if video_path is None:
            for c in input_path.iterdir():
                if c.is_file() and c.suffix.lower() in (".mp4", ".avi"):
                    video_path = c
                    break
        if video_path is None:
            raise FileNotFoundError(f"No video file found in folder {input_path}")

        timestamps_path = input_path / "frame_timestamps.csv"
        timestamps: list[float] | None = None
        if timestamps_path.exists():
            timestamps = []
            try:
                with open(timestamps_path, "r") as f:
                    for i, row in enumerate(f):
                        row = row.strip()
                        if not row:
                            continue
                        if i == 0 and "frame_index" in row:
                            continue
                        parts = row.split(",")
                        if len(parts) < 2:
                            continue
                        try:
                            timestamps.append(float(parts[1]))
                        except Exception:
                            continue
            except Exception:
                timestamps = None

        return str(video_path), timestamps

    @staticmethod
    def _resolve_sidecar(search_dir: Path | None, name: str, default_path: Path) -> Path | None:
        candidates = []
        if search_dir is not None:
            candidates.append(search_dir / name)
        candidates.append(default_path)
        for path in candidates:
            if path.is_file():
                return path
        return None

    @staticmethod
    def _build_camera_model_from_sidecars(camera_json_path: Path, camera_position_path: Path):
        """Create CameraModel from sidecars; optional third angle is mount yaw.

        Legacy two-angle files retain the historical +90-degree yaw. The
        optional third angle is the physical mount yaw in the robot-frame
        convention; CameraModel uses the opposite rotation convention.
        """
        if CameraModel is None:
            print("CameraModel import failed; cannot compute rays")
            return None

        if not camera_json_path.is_file():
            raise FileNotFoundError(f"Camera intrinsics file not found: {camera_json_path}")
        if not camera_position_path.is_file():
            raise FileNotFoundError(f"Camera position file not found: {camera_position_path}")

        try:
            with open(camera_position_path, "r") as cpf:
                cam_pos_line = cpf.readline().strip()
                cam_orient_line = cpf.readline().strip()
            cam_x, cam_y, cam_z = map(float, cam_pos_line.split())
            camera_angles = list(map(float, cam_orient_line.split()))
            if len(camera_angles) not in (2, 3):
                raise ValueError("camera_position orientation must be roll pitch [mount_yaw] in radians")
            cam_x_angle, cam_y_angle = camera_angles[:2]
            # Gazebo/SDF cameras look along their local +X axis; CameraModel
            # defines its zero-tilt optical axis as +Z.  ``camera_up`` has
            # SDF pitch=-pi/2, which therefore already describes a vertical
            # camera and must not be applied as a CameraModel pitch.
            camera_spec = json.loads(camera_json_path.read_text(encoding="utf-8"))
            is_gazebo_sdf = any("gazebo_sdf" in str(key) for key in camera_spec)
            if is_gazebo_sdf:
                sdf_yaw = camera_angles[2]
                camera_tilts = [0.0, 0.0, sdf_yaw - np.pi / 2.0]
            else:
                # camera.config describes robot->camera mount yaw, whereas
                # CameraModel rotates camera rays into robot axes.  These are
                # inverse rotations.  Passing the physical -90 degrees
                # directly here flipped every horizontal ray by 180 degrees
                # relative to the proven legacy +90-degree calibration.
                cam_yaw_angle = -camera_angles[2] if len(camera_angles) == 3 else np.deg2rad(90)
                camera_tilts = [-cam_x_angle, cam_y_angle, cam_yaw_angle]
            print(f"Using camera.json: {camera_json_path}")
            print(f"Using camera_position.txt: {camera_position_path}")
            print(f"Camera mount: position_cm={[cam_x, cam_y, cam_z]}, angles_rad={camera_angles}")
            return CameraModel(
                offsets=[cam_x / 100, cam_y / 100, cam_z / 100],
                tilts_rad=camera_tilts,
                camera_json_path=str(camera_json_path),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CameraModel from explicit sidecars: {e}") from e

    @staticmethod
    def _precompute_camera_origin(camera_model):
        if camera_model is None:
            return None
        try:
            origin, _ = camera_model.pixels_to_rays(
                np.array([0.0, 0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                np.array([[0.0, 0.0]], dtype=np.float64),
            )
            return [float(v) for v in origin]
        except Exception:
            return None

    @line_profiler.profile
    def process_frame(self):
        ret, self.image = self.cap.read()
        if not ret:
            return False, None, None

        if self.frame_timestamps is not None:
            if self.frame_index < len(self.frame_timestamps):
                self.current_frame_timestamp = float(self.frame_timestamps[self.frame_index])
            elif self.frame_timestamps:
                self.current_frame_timestamp = float(self.frame_timestamps[-1])
            else:
                self.current_frame_timestamp = None
        else:
            self.current_frame_timestamp = None

        dets = self.detector(self.image, verbose=False)[0].boxes.data.cpu().numpy()
        # print(dets)
        # raise Exception("Stop")
        # Feed every detector box to BoT-SORT so a temporarily clipped marker
        # does not break its persistent track ID.  Only observations exported
        # for decoding/ray construction are filtered below.
        tracks = self.tracker.update(dets, self.image)
        tracks = self._filter_track_observations(tracks, self.image.shape)
        # print(tracks)
        self.process_ceil_labels_fourier(tracks, self.image, (None, None) if self.current_coordinates is None else self.current_coordinates)
        self.current_frame_camera_meta = self._compute_frame_camera_meta(tracks)
        self.frame_index += 1
        return True, tracks, self.current_frame_timestamp

    @classmethod
    def _filter_track_observations(cls, tracks: list[STrack], image_shape) -> list[STrack]:
        """Reject unreliable observations without interrupting the tracker.

        This intentionally mirrors ``get_video_data_detections.py``.  The
        full detection stream has already updated BoT-SORT; this method only
        controls which current boxes may produce decoded data and camera rays.
        """
        if not tracks:
            return []

        height, width = image_shape[:2]
        records = []
        for track in tracks:
            x1, y1, x2, y2 = map(float, track.tlbr)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            area = box_width * box_height
            longest_side = max(box_width, box_height)
            aspect = min(box_width, box_height) / longest_side if longest_side > 0 else 0.0
            clipped = (
                x1 <= cls.EDGE_EPSILON_PX
                or y1 <= cls.EDGE_EPSILON_PX
                or x2 >= width - cls.EDGE_EPSILON_PX
                or y2 >= height - cls.EDGE_EPSILON_PX
            )
            records.append((track, area, aspect, clipped))

        positive = np.asarray([area for _, area, _, _ in records if area > 0], dtype=float)
        if len(positive) >= 3:
            log_area = np.log(positive)
            center = float(np.median(log_area))
            mad = float(np.median(np.abs(log_area - center)))
            robust_sigma = max(1.4826 * mad, 0.05)
            min_area = float(np.exp(center - cls.ROBUST_SIZE_SIGMA * robust_sigma))
        else:
            min_area = 0.0

        return [
            track
            for track, area, aspect, clipped in records
            if not clipped and aspect >= cls.MIN_ASPECT_RATIO and area >= min_area
        ]

    def _compute_frame_camera_meta(self, tracks: list[STrack]) -> dict | None:
        """Return per-track center/ray metadata keyed by track_id."""
        if self.camera_model_for_export is None:
            return None

        centers: list[list[float]] = []
        track_ids: list[int] = []
        for track in tracks or []:
            center = getattr(track, "center", None)
            if center is None:
                # BoT-SORT exposes the detector/image convention through
                # ``tlbr``: [x1, y1, x2, y2].  ``ltrb`` is the legacy
                # transposed view [y1, x1, y2, x2] and must never be used for
                # pixels or camera-ray back-projection.
                l, t, r, b = list(track.tlbr)
                center = ((float(l) + float(r)) / 2.0, (float(t) + float(b)) / 2.0)
            # CameraModel expects image pixels as [u=x, v=y]. Keep this
            # identical to the exported ``center`` and legacy bbox midpoint.
            centers.append([float(center[0]), float(center[1])])
            track_ids.append(int(track.track_id))

        rays: list[list[float]] = []
        if centers:
            try:
                _, rays_np = self.camera_model_for_export.pixels_to_rays(
                    np.array([0.0, 0.0], dtype=np.float64),
                    np.array([0.0], dtype=np.float64),
                    np.asarray(centers, dtype=np.float64),
                )
                rays = [[float(v) for v in ray] for ray in rays_np]
            except Exception:
                rays = []

        track_meta: dict[int, dict[str, list[float] | None]] = {}
        for i, tid in enumerate(track_ids):
            ray_i = rays[i] if i < len(rays) else None
            track_meta[tid] = {
                "center": centers[i],
                "ray": ray_i,
            }
        return track_meta

    def process_ceil_labels_fourier(self, tracks: list[STrack], im: npt.NDArray, coord_prev: tuple[float | None, float | None]):
        if not tracks:
            return
        dets = [track.dets for track in tracks]
        if self.fourier_decoder is None:
            for det, track in zip(dets, tracks):
                center = (int((det[0] + det[2]) // 2), int((det[1] + det[3]) // 2))
                track.update_code_and_center([-1], [1.0], center)
            self.prev_codes = []
            return
        crops = [im[int(det[1]):int(det[3]), int(det[0]):int(det[2])] for det in dets]

        # Build allowed codes per image if adjacency available and previous codes exist
        allowed_codes_per_image: list[list[int] | None] = []
        use_adjacency = bool(self.code_adjacency)
        for track in tracks:
            prev_codes = self.prev_codes
            if use_adjacency and prev_codes:
                allowed = set()
                for code in prev_codes:
                    neighbors = self.code_adjacency.get(code)
                    if neighbors:
                        allowed.update(neighbors)
                # Also include the previous codes themselves
                allowed.update(prev_codes)
                allowed_codes_per_image.append(list(allowed))
            else:
                allowed_codes_per_image.append(None)

        if coord_prev[0] is not None:
            res = self.fourier_decoder.predict_with_labels(crops, point = coord_prev, radius = 1000)
        else:
            res = self.fourier_decoder.predict_with_labels(crops)
        # codes = [lbl[0] for lbl in res[0]]
        thresh = self.args.fourier_confidence_thresh
        codes_and_confs = [(r, conf) if (len(conf) >= 1 and conf[0] > thresh) else ([-2],  [0]) for r, conf in zip(res[0], res[1])]

        centers = [(int((det[0] + det[2]) // 2), int((det[1] + det[3]) // 2)) for det in dets]
        raw_decoder_enabled = bool(getattr(self.args, "raw_decoder", False))
        for raw_codes, raw_confs, code_and_conf, center, track in zip(res[0], res[1], codes_and_confs, centers, tracks):
            track.update_code_and_center(code_and_conf[0], code_and_conf[1], center)
            if raw_decoder_enabled:
                # Keep per-frame decoder candidates for optional raw JSON export.
                track.raw_codes = _to_python_list(raw_codes, int)
                track.raw_confs = _to_python_list(raw_confs, float)
        
        # Get codes from all tracks at the frame, store for next frame
        self.prev_codes = [track.code for track in tracks]

    @line_profiler.profile
    @staticmethod
    def process_frame_data(targets: list[STrack], raw_decoder: bool = False, frame_camera_meta: dict | None = None):
        data = dict()
        if not targets:
            return data
        for target in targets:
            # Optimize tensor-to-list conversion
            if isinstance(target.code, torch.Tensor):
                # Method 1: More efficient conversion without modifying original
                if target.code.is_cuda:
                    # Move to CPU first, then convert (faster than direct tolist() on CUDA)
                    code = target.code.cpu().numpy().tolist()
                else:
                    # Use numpy intermediate for CPU tensors (faster than direct tolist())
                    code = target.code.numpy().tolist()
            else:
                code = target.code

            record = {"tlbr": list(target.tlbr)}
            if raw_decoder:
                record["codes"] = _to_python_list(getattr(target, "raw_codes", []), int)
                record["confs"] = _to_python_list(getattr(target, "raw_confs", []), float)
            else:
                record["code"] = code

            center = getattr(target, "center", None)
            if center is None:
                l, t, r, b = list(target.tlbr)
                center = ((float(l) + float(r)) / 2.0, (float(t) + float(b)) / 2.0)
            record["center"] = [float(center[0]), float(center[1])]

            ray = None
            tid = int(target.track_id)
            if frame_camera_meta is not None:
                meta = frame_camera_meta.get(tid)
                if isinstance(meta, dict):
                    ray = meta.get("ray")
                    center_meta = meta.get("center")
                    if center_meta is not None:
                        record["center"] = [float(center_meta[0]), float(center_meta[1])]
            record["ray"] = ray

            data[target.track_id] = record
        return data

    def start(self, save: bool = True):
        video_data = list()
        frame_timestamps: list[float | None] = []

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total_frames if total_frames > 0 else None)
        while True:
            ret, targets, timestamp = self.process_frame()
            if not ret:
                break
            video_data.append(
                self.process_frame_data(
                    targets,
                    raw_decoder=bool(getattr(self.args, "raw_decoder", False)),
                    frame_camera_meta=self.current_frame_camera_meta,
                )
            )
            frame_timestamps.append(timestamp)
            pbar.update(1)
        pbar.close()

        if save:
            self._save_output(video_data, frame_timestamps)
        return video_data

    def _save_output(self, video_data: list[dict], frame_timestamps: list[float | None]) -> None:
        if self.source_dir is None:
            save_video_data(video_data, self.video_path)
            return

        payload: dict[str, object] = {
            "video_path": str(self.video_path),
            "data": video_data,
            "frame_timestamps": frame_timestamps,
            "camera_origin": self.camera_origin_for_export,
            "ray_frame": "CameraModel world axes at robot_pos=[0,0], robot_orient=[0]; rotates with robot body yaw during map projection",
            "camera_origin_frame": "CameraModel robot pose zero; camera mount relative to rotation axis",
            "ray_source": "CameraModel.pixels_to_rays from detection pixel center and recording camera sidecars",
        }

        out_path = self.source_dir / "observations.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Data saved to {out_path}")


if __name__ == "__main__":
    # parser1 = argparse.ArgumentParser(description="Обрабатывает видео и делает json с бибоксами и номерами меток")
    # parser1.add_argument("file_path", type=str, help="Путь до файла")
    # args_script = parser1.parse_args()
    # tracker_args.video = "/mnt/Data/Videos/Вешки навигация/x5/x5_2025_04_10_16_56_36.mp4"
    video_processor = VideoProcessor(
        tracker_args.video, str(AUTO_MAP_DIR / "model.onnx"), tracker_args,
        labels=tracker_args.labels
    )
    video_processor.start()
