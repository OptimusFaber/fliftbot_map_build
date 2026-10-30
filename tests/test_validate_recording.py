import sys
import json
import subprocess
import tempfile
import numpy as np
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/map_build"))

from validate_recording import moving_encoder_gaps


class ValidateRecordingTest(unittest.TestCase):
    def test_rejects_empty_encoder_and_nonfinite_lidar_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'horizon').mkdir()
            np.savez(root / 'horizon/00001.npz', stamp=np.nan, xy=np.zeros((3, 2)))
            (root / 'encoder.jsonl').write_text('')
            result = subprocess.run([sys.executable, str(ROOT / 'src/map_build/validate_recording.py'),
                                     str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            errors = json.loads(result.stdout)['errors']
            self.assertTrue(any('invalid NPZ' in error for error in errors))
            self.assertTrue(any('no valid samples' in error for error in errors))

    def test_rejects_unrecoverable_moving_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'horizon').mkdir()
            for index, stamp in enumerate([1., 3.]):
                np.savez(root / f'horizon/{index:05d}.npz', stamp=stamp, xy=np.zeros((3, 2)))
            (root / 'encoder.jsonl').write_text(''.join(json.dumps(
                dict(enc_unix=t, enc_v=.5, enc_dgr=0))+'\n' for t in [1., 3.]))
            result = subprocess.run([sys.executable, str(ROOT / 'src/map_build/validate_recording.py'),
                                     str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('exceeds repair limit', result.stdout)

    def test_reports_only_moving_gaps(self):
        rows = [
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.1, 0.3, 0.0),
            (3.15, 0.3, 0.0),
        ]
        gaps = moving_encoder_gaps(rows, 0.2)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0], 1.05)


if __name__ == "__main__":
    unittest.main()
