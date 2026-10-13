#!/usr/bin/env python3
"""Marker processing entry point; delegates to the vendored evaluator."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--map-dir',type=Path,required=True);p.add_argument('--camera-position',type=Path,required=True);p.add_argument('--lidar-config',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--ceiling-z',type=float);a=p.parse_args()
 cmd=[sys.executable,str(ROOT/'vendor/fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_stored_ray_corrected.py'),'--observations',str(a.observations),'--trajectory',str(a.trajectory),'--map-dir',str(a.map_dir),'--camera-position',str(a.camera_position),'--lidar-config',str(a.lidar_config),'--out-dir',str(a.out_dir)]
 if a.ceiling_z is not None:cmd += ['--ceiling-z',str(a.ceiling_z)]
 return subprocess.call(cmd)
if __name__=='__main__':raise SystemExit(main())
