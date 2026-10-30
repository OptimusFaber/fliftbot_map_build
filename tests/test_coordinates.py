import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "map_build"


class CoordinatePipelineTest(unittest.TestCase):
    def test_marker_export_preserves_unknown_height(self):
        sys.path.insert(0, str(SRC))
        from build_from_canonical import export_marker_coordinates
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'web_map').mkdir()
            (root / 'markers').mkdir()
            np.save(root / 'web_map/pbstream_to_local_matrix.npy',
                    np.array([[1, 0, -2], [0, 1, -3], [0, 0, 1]]))
            np.save(root / 'web_map/pbstream_to_raster_matrix.npy',
                    np.array([[10, 0, 100], [0, -10, 100], [0, 0, 1]]))
            (root / 'markers/marker_points_stored_ray_corrected.json').write_text(json.dumps({
                'status': 'accepted', 'markers': [{'marker_id': 'marker_001', 'point_xyz_m': [2, 3, None]}]}))
            export_marker_coordinates(root)
            marker = json.loads((root / 'markers/markers_map.json').read_text())['markers'][0]
            self.assertEqual(marker['local_map_xyz_m'], [0, 0, None])
            self.assertEqual(marker['web_raster_uv_px'], [120, 70])
            with (root / 'markers/markers_map.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['height_m'], '')
            self.assertEqual(float(row['raster_v_px']), 70)

    def test_web_matrices_and_origin_invariant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir, web_dir = root / "map", root / "web"
            map_dir.mkdir()
            image = np.full((120, 160), 128, dtype=np.uint8)
            image[30:34, 35:125] = 0
            image[85:89, 35:125] = 0
            image[30:89, 35:39] = 0
            image[30:89, 121:125] = 0
            Image.fromarray(image).save(map_dir / "map.pgm")
            (map_dir / "map.yaml").write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [-4.0, -3.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n", encoding="utf-8")
            subprocess.run([sys.executable, str(SRC / "build_web_map.py"), "--map-dir", str(map_dir),
                            "--output", str(web_dir), "--ray-filter-min-cells", "1"], check=True)
            local_to = np.load(web_dir / "map_transform_matrix.npy")
            pb_to_local = np.load(web_dir / "pbstream_to_local_matrix.npy")
            pb_to_raster = np.load(web_dir / "pbstream_to_raster_matrix.npy")
            np.testing.assert_allclose(local_to @ pb_to_local, pb_to_raster, atol=1e-9)
            before = pb_to_raster.copy()
            subprocess.run([sys.executable, str(SRC / "set_map_origin.py"), "--web-map", str(web_dir),
                            "--origin-map-xy", "1.25", "-0.75", "--yaw-deg", "17"], check=True)
            local_to = np.load(web_dir / "map_transform_matrix.npy")
            pb_to_local = np.load(web_dir / "pbstream_to_local_matrix.npy")
            pb_to_raster = np.load(web_dir / "pbstream_to_raster_matrix.npy")
            np.testing.assert_allclose(pb_to_raster, before, atol=1e-12)
            np.testing.assert_allclose(local_to @ pb_to_local, pb_to_raster, atol=1e-9)
            np.testing.assert_allclose(pb_to_local @ [1.25, -0.75, 1.0], [0, 0, 1], atol=1e-9)
            metadata = json.loads((web_dir / "map_metadata.json").read_text())
            self.assertTrue(metadata["origin"]["image_unchanged"])


if __name__ == "__main__":
    unittest.main()
