import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/map_build"))

from convert_recording import detect_odometry_kind
from run_pipeline import read_camera_position_file, wheel_odometry_trajectory


class PipelineOptionsTest(unittest.TestCase):
    def test_odometry_kind_uses_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "wrong.csv"
            log.write_text("enc_t,enc_v,enc_dgr\n2026_01_01_00_00_00_000,1,2\n")
            encoder = root / "wrong.log"
            encoder.write_text("timestamp_sec,linear_velocity_m_per_s,wheel_angle_rad\n1,2,3\n")
            self.assertEqual(detect_odometry_kind(log), "log")
            self.assertEqual(detect_odometry_kind(encoder), "csv")

    def test_camera_position_file_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.txt"
            path.write_text("107.7 -27.4 124.8\n0.1 -0.2 -1.5707963267948966\n")
            position, orientation = read_camera_position_file(path)
            self.assertAlmostEqual(position["x"], 1.077)
            self.assertAlmostEqual(position["y"], -0.274)
            self.assertAlmostEqual(position["z"], 1.248)
            self.assertAlmostEqual(orientation["yaw"], -90.0)

    def test_wheel_odometry_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder.jsonl"
            encoder.write_text(
                '{"enc_unix":1.0,"enc_v":1.0,"enc_dgr":0.0}\n'
                '{"enc_unix":1.1,"enc_v":1.0,"enc_dgr":0.0}\n')
            output = root / "trajectory.json"
            wheel_odometry_trajectory(encoder, output, {"wheelbase_m": 1.39}, 1.0)
            import json
            result = json.loads(output.read_text())
            self.assertEqual(result["tracking_frame"], "rotation_axis")
            self.assertAlmostEqual(result["poses"][-1]["x"], 0.1)

    def test_wheel_odometry_repairs_short_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder.jsonl"
            encoder.write_text(
                '{"enc_unix":1.0,"enc_v":1.0,"enc_dgr":0.0}\n'
                '{"enc_unix":2.05,"enc_v":1.0,"enc_dgr":0.0}\n')
            output = root / "trajectory.json"
            wheel_odometry_trajectory(encoder, output, {"wheelbase_m": 1.39}, 1.0)
            import json
            result = json.loads(output.read_text())
            self.assertAlmostEqual(result["poses"][-1]["x"], 1.05)
            self.assertEqual(result["encoder_gap_repair"]["repaired_gaps"], 1)
            self.assertGreater(result["encoder_gap_repair"]["inserted_samples"], 0)


if __name__ == "__main__":
    unittest.main()
