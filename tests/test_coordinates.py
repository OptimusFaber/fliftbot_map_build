import json
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
