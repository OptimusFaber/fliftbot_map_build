"""Regression checks for simulator import and calibrated navigation deployment."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / 'src/map_build'
sys.path.insert(0, str(SRC))
from import_sim_recording import import_recording


class NavigationHandoff(unittest.TestCase):
    def test_sim_import_excludes_gt_and_preserves_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'sim/lidar'
            (source / 'horizon').mkdir(parents=True)
            np.savez(source / 'horizon/00001.npz', xy=np.zeros((120, 2), dtype=np.float32),
                     stamp=np.asarray(120., dtype=np.float64), t=np.full(120, 120.), frame=np.asarray(1, dtype=np.int32))
            config = {'format': 'lidar.config.v1', 'lidar_type': 'dephan_m360',
                      'source': 'simulation_robot_geometry',
                      'position_m': {'x': 1.37, 'y': .315, 'z': 2.04},
                      'orientation_deg': {'roll': 0, 'pitch': 0, 'yaw': 0}}
            (source / 'lidar.config').write_text(json.dumps(config))
            encoder = {'enc_unix': 120., 'enc_v': -.2, 'enc_dgr': 30,
                       'odom_x': 999, 'odom_y': 999, 'odom_yaw': 2, 'enc_wz': 100}
            original = json.dumps(encoder) + '\n'
            (source / 'encoder.jsonl').write_text(original)
            (source / 'odom.json').write_text('GT must never enter mapping')
            output = root / 'canonical'
            import_recording(source.parent, output)
            row = json.loads((output / 'encoder.jsonl').read_text())
            self.assertEqual(set(row), {'enc_unix', 'enc_v', 'enc_dgr', 'enc_available'})
            self.assertFalse((output / 'odom.json').exists())
            self.assertEqual((source / 'encoder.jsonl').read_text(), original)
            lidar = json.loads((output / 'lidar.config').read_text())
            self.assertEqual(lidar['schema'], 'fliftbot.component_config.v1')
            self.assertEqual(lidar['encoder_speed_sign'], -1)
            self.assertEqual(lidar['mapping_orientation_deg']['yaw'], 0)
            robot = json.loads((output / 'liftbot.config').read_text())
            self.assertEqual(robot['wheelbase_m'], 1.410)
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                import_recording(source.parent, output)

    def test_bundle_carries_effective_calibration_and_correct_inferred_yaw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'run', root / 'deploy'
            (source / 'map').mkdir(parents=True)
            (source / 'web_map').mkdir()
            (source / 'map/map.pbstream').write_bytes(b'unit-fixture')
            image = Image.fromarray(np.zeros((30, 40), dtype=np.uint8))
            image.save(source / 'map/map.pgm')
            image.save(source / 'map/map.png')
            (source / 'map/map.yaml').write_text('image: map.pgm\nresolution: 0.05\norigin: [0,0,0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
            (source / 'web_map/map_metadata.json').write_text(json.dumps({'origin': {'pbstream_map_xy_m': [1, 2]}}))
            angle = math.radians(17)
            c, s = math.cos(angle), math.sin(angle)
            matrix = np.array([[c, s, -c-2*s], [-s, c, s-2*c], [0,0,1]])
            np.save(source / 'web_map/pbstream_to_local_matrix.npy', matrix)
            parameters = {'lidar': {'lidar_type': 'dephan_m360', 'max_range_m': 40,
                                    'encoder_speed_sign': -1, 'position_m': {'x': 1.111, 'y': .222},
                                    'orientation_deg': {'roll': 0, 'pitch': 0, 'yaw': 180}},
                          'robot': {'wheelbase_m': 1.46, 'steering_sign': -1, 'steering_offset_deg': 5}}
            (source / 'run_parameters.json').write_text(json.dumps(parameters))
            command = [sys.executable, str(SRC / 'package_bundle.py'), '--run', str(source), '--output', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            reference = json.loads((output / 'origin_reference.json').read_text())
            self.assertAlmostEqual(reference['origin_map_yaw_rad'], angle)
            config = json.loads((output / 'navigation_config.json').read_text())
            self.assertEqual(config['scan_to_base']['x_m'], 1.111)
            self.assertEqual(config['odometry']['wheelbase_m'], 1.46)
            self.assertEqual(config['odometry']['steering_zero_deg'], 5)
            self.assertEqual(config['odometry']['model'], 'drive_wheel')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
