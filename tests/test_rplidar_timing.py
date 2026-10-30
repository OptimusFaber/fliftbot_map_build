import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/realtime_mapping_scripts'))
from rplidar_to_run import RAW_MAGIC, RAW_RECORD, iter_raw_records, write_horizon


class RplidarTimingTest(unittest.TestCase):
    def test_revolutions_have_ordered_nonoverlapping_beam_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'scan.rplidar'
            with source.open('wb') as stream:
                stream.write(RAW_MAGIC)
                for index in range(4 * 360):
                    angle = -math.pi + (index % 360) * math.tau / 360
                    device = 10 + index * .1 / 360
                    stream.write(RAW_RECORD.pack(1000 + device, math.cos(angle), math.sin(angle), device, 0))
            report = write_horizon(source, root / 'horizon')
            scans = sorted((root / 'horizon').glob('*.npz'))
            self.assertEqual(len(scans), 4)
            self.assertEqual(report['frames'], 4)
            previous_end = -math.inf
            for path in scans:
                with np.load(path) as data:
                    self.assertTrue(np.all(np.diff(data['t']) > 0))
                    self.assertGreater(data['t'][0], previous_end)
                    self.assertAlmostEqual(float(data['stamp']), data['t'][-1])
                    previous_end = data['t'][-1]
            # Run all Python entry-point wrappers from raw to materialized mapping inputs.
            encoder = root / 'wheel.csv'
            encoder.write_text('timestamp_sec,linear_velocity_m_per_s,wheel_angle_rad\n' +
                               ''.join(f'{t},0.2,0\n' for t in np.arange(1009.8, 1010.6, .02)))
            command = [sys.executable, 'src/map_build/build_from_raw.py', '--scan', str(source),
                       '--odometry', str(encoder), '--canonical-output', str(root / 'canonical'),
                       '--lidar-config', 'configs/bs_rplidar/lidar.config',
                       '--camera-config', 'configs/bs_rplidar/camera.config',
                       '--robot-config', 'configs/bs_rplidar/liftbot.config',
                       '--output', str(root / 'results'), '--name', 'profile_forwarding',
                       '--marker-profile', 'test_site', '--marker-time-offset', '.25',
                       '--fork-position', 'lowered', '--dry-run']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = list((root / 'results').glob('*/run_parameters.json'))
            self.assertEqual(len(outputs), 1)
            params = json.loads(outputs[0].read_text())
            self.assertEqual(params['markers']['profile']['name'], 'test_site')
            self.assertEqual(params['markers']['min_camera_baseline_m'], .25)
            self.assertEqual(params['markers']['time_offset_s'], .25)
            self.assertAlmostEqual(params['robot']['wheelbase_m'], 1.46)
            # Reusing the same canonical directory must fail instead of mixing old packets.
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('canonical output already contains data', result.stderr)
            with source.open('ab') as stream:
                stream.write(b'broken')
            with self.assertRaises(ValueError):
                list(iter_raw_records(source))


if __name__ == '__main__':
    unittest.main()
