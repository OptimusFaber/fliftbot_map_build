"""Same observation export as get_video_data.py, but boxes come from YOLO, not BoT-SORT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy import typing as npt

from get_video_data import (
    AUTO_MAP_DIR,
    VideoProcessor,
    _to_python_list,
)
from utils.tracker.tracking_utils.args import args as tracker_args
from utils.utils import save_video_data


class DetectionBox:
    """Minimal stand-in for STrack so Fourier decode and JSON export stay the same."""

    def __init__(self, det_id: int, xyxy: npt.NDArray, score: float):
        x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
        self.track_id = det_id
        self.dets = np.array([x1, y1, x2, y2], dtype=float)
        self.tlbr = self.dets.copy()
        self.score = float(score)
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.code = -2
        self.raw_codes: list[int] = []
        self.raw_confs: list[float] = []

    def update_code_and_center(self, code_list, conf_list, center, decay_rate=0.95):
        del decay_rate
        self.code = code_list[0] if len(code_list) else -2
        self.center = center


class DetectionVideoProcessor(VideoProcessor):
    # These are shape/robustness limits, not an assumed physical pixel size.
    # The minimum area is estimated independently in every frame from the
    # current detector population.
    MIN_ASPECT_RATIO = 0.72
    EDGE_EPSILON_PX = 1.0
    ROBUST_SIZE_SIGMA = 3.0

    def __init__(self, video_path, model_path, bs_args, labels):
        super().__init__(video_path, model_path, bs_args, labels)
        self._next_det_id = 1
        self.conf_thresh = float(getattr(bs_args, "track_low_thresh", 0.1))

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

        raw = self.detector(self.image, verbose=False)[0].boxes.data.cpu().numpy()
        detections = self._boxes_from_yolo(raw)
        detections = self._filter_detection_boxes(detections)
        self.process_ceil_labels_fourier(
            detections,
            self.image,
            (None, None) if self.current_coordinates is None else self.current_coordinates,
        )
        self.current_frame_camera_meta = self._compute_frame_camera_meta(detections)
        self.frame_index += 1
        return True, detections, self.current_frame_timestamp

    def _boxes_from_yolo(self, raw: npt.NDArray) -> list[DetectionBox]:
        if raw is None or len(raw) == 0:
            return []
        detections = []
        for row in raw:
            score = float(row[4]) if row.shape[0] >= 5 else 1.0
            if score < self.conf_thresh:
                continue
            det_id = self._next_det_id
            self._next_det_id += 1
            detections.append(DetectionBox(det_id, row[:4], score))
        return detections

    def _filter_detection_boxes(self, detections: list[DetectionBox]) -> list[DetectionBox]:
        """Reject clipped/non-square boxes and adaptive small-box outliers.

        YOLO detections in this exporter are deliberately not tracked across
        frames, so the reference size is the robust distribution of boxes in
        the current frame.  This avoids baking in a 50/58-pixel assumption and
        adapts to camera distance and image scale.
        """
        if not detections:
            return []
        height, width = self.image.shape[:2]
        records = []
        for det in detections:
            x1, y1, x2, y2 = map(float, det.tlbr)
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            area = w * h
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
            clipped = (x1 <= self.EDGE_EPSILON_PX or y1 <= self.EDGE_EPSILON_PX or
                       x2 >= width - self.EDGE_EPSILON_PX or y2 >= height - self.EDGE_EPSILON_PX)
            records.append((det, area, aspect, clipped))

        positive = np.asarray([area for _, area, _, _ in records if area > 0], dtype=float)
        if len(positive) >= 3:
            log_area = np.log(positive)
            center = float(np.median(log_area))
            mad = float(np.median(np.abs(log_area - center)))
            robust_sigma = max(1.4826 * mad, 0.05)
            min_area = float(np.exp(center - self.ROBUST_SIZE_SIGMA * robust_sigma))
        else:
            min_area = 0.0

        kept = []
        for det, area, aspect, clipped in records:
            # A box touching an image edge can contain only a partial marker.
            # A strongly rectangular box is independently rejected as well.
            if clipped or aspect < self.MIN_ASPECT_RATIO or area < min_area:
                continue
            kept.append(det)
        return kept

    def _compute_frame_camera_meta(self, tracks: list[DetectionBox]) -> dict | None:
        if self.camera_model_for_export is None:
            return None

        centers: list[list[float]] = []
        track_ids: list[int] = []
        for det in tracks or []:
            centers.append([float(det.center[0]), float(det.center[1])])
            track_ids.append(int(det.track_id))

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
            track_meta[tid] = {
                "center": centers[i],
                "ray": rays[i] if i < len(rays) else None,
            }
        return track_meta

    @staticmethod
    def process_frame_data(targets: list[DetectionBox], raw_decoder: bool = False, frame_camera_meta: dict | None = None):
        data = dict()
        if not targets:
            return data
        for target in targets:
            record = {
                "tlbr": [float(v) for v in target.tlbr],
                "score": float(target.score),
                "box_source": "detection",
            }
            if raw_decoder:
                record["codes"] = _to_python_list(getattr(target, "raw_codes", []), int)
                record["confs"] = _to_python_list(getattr(target, "raw_confs", []), float)
            else:
                record["code"] = target.code

            center = target.center
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

    def _save_output(self, video_data: list[dict], frame_timestamps: list[float | None]) -> None:
        if self.source_dir is None:
            save_video_data(video_data, Path(self.video_path).with_name(Path(self.video_path).stem + "_detections"))
            return

        payload: dict[str, object] = {
            "video_path": str(self.video_path),
            "data": video_data,
            "frame_timestamps": frame_timestamps,
            "camera_origin": self.camera_origin_for_export,
            "ray_frame": "CameraModel world axes at robot_pos=[0,0], robot_orient=[0]; rotates with robot body yaw during map projection",
            "camera_origin_frame": "CameraModel robot pose zero; camera mount relative to rotation axis",
            "ray_source": "CameraModel.pixels_to_rays from YOLO detection pixel center and recording camera sidecars",
            "box_source": "yolo_detection",
        }

        out_path = self.source_dir / "observations_detections.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Data saved to {out_path}")


if __name__ == "__main__":
    video_processor = DetectionVideoProcessor(
        tracker_args.video, str(AUTO_MAP_DIR / "model.onnx"), tracker_args,
        labels=tracker_args.labels
    )
    video_processor.start()
