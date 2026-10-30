import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/map_build'))
from marker_profiles import add_marker_options, resolve_marker_parameters


class MarkerProfilesTest(unittest.TestCase):
    def parse(self, *options):
        parser = argparse.ArgumentParser()
        add_marker_options(parser)
        return parser.parse_args(options)

    def test_override_and_fixed_plane(self):
        params, profile = resolve_marker_parameters(self.parse(
            '--marker-profile', 'test_site', '--ceiling-z', '3.2', '--merge-radius-m', '.2'))
        self.assertFalse(params['per_track_height'])
        self.assertEqual(params['merge_radius_m'], .2)
        self.assertEqual(profile['name'], 'test_site')
        with self.assertRaises(ValueError):
            resolve_marker_parameters(self.parse('--block-distance-m', '0'))
        with self.assertRaises(ValueError):
            resolve_marker_parameters(self.parse('--ceiling-z', '3', '--per-track-height'))

    def test_short_low_marker_tracks_and_stationary_abstention(self):
        # Exercise the real CLI, geometry, profile resolution and JSON output.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'map').mkdir()
            (run / 'input').mkdir()
            Image.new('L', (100, 100), 254).save(run / 'map/map.pgm')
            (run / 'map/map.yaml').write_text('image: map.pgm\nresolution: 0.1\norigin: [-5, -5, 0]\n')
            (run / 'input/camera_position.txt').write_text('0 0 100\n0 0 0\n')
            (run / 'input/camera.json').write_text((ROOT / 'configs/camera_intrinsics.json').read_text())
            (run / 'input/lidar.markers.config').write_text(json.dumps({
                'position_m': dict(x=0, y=0, z=0), 'orientation_deg': dict(yaw=0),
                'ray_correction_deg': dict(roll=0, pitch=0, yaw=0)}))
            targets = np.array([[0, 0, 3], [2, 0, 3], [0, 2, 3]], float)
            times = 1000 + np.arange(41) * .04
            observations, poses = [], []
            for index, (stamp, x) in enumerate(zip(times, np.linspace(-.2, .2, len(times)))):
                frame = {}
                for track, target in enumerate(targets):
                    ray = target - [x, 0, 1]
                    frame[str(track)] = {'ray': (ray / np.linalg.norm(ray)).tolist(), 'code': track}
                observations.append(frame)
                poses.append(dict(timestamp_universal=621355968000000000 + round(stamp * 1e7),
                                  x=x, y=0, qz=0, qw=1))
            obs_path = run / 'observations.json'
            obs_path.write_text(json.dumps(dict(data=observations, frame_timestamps=times.tolist())))
            (run / 'map/trajectory.json').write_text(json.dumps(dict(tracking_frame='lidar', poses=poses)))
            # The standalone reprocessor must use the same camera offset stored by the full pipeline.
            (run / 'run_parameters.json').write_text(json.dumps(dict(markers=dict(time_offset_s=0))))
            for profile, expected in [('warehouse', 0), ('test_site', 3)]:
                output = run / profile
                command = [sys.executable, str(ROOT / 'src/map_build/process_markers.py'),
                           '--run-dir', str(run), '--out-dir', str(output), '--marker-profile', profile]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((output / 'marker_points_stored_ray_corrected.json').read_text())
                self.assertEqual(len(report['markers']), expected)
                for marker in report['markers']:
                    self.assertAlmostEqual(marker['ceiling_z_m'], 3, places=6)
                    self.assertEqual(marker['height_status'], 'estimated')
                params = json.loads((output / 'run_parameters.json').read_text())
                self.assertEqual(params['profile']['name'], profile)
                if profile == 'test_site':
                    # Deliberately demand more independent positions than are available.
                    # Stable least-squares heights alone must not bypass the motion gate.
                    result = subprocess.run(command + ['--stationary-min-groups', '100'],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads((output / 'marker_points_stored_ray_corrected.json').read_text())
                    self.assertEqual(len(report['markers']), 3)
                    self.assertTrue(all(m['ceiling_z_m'] is None for m in report['markers']))


if __name__ == '__main__':
    unittest.main()
