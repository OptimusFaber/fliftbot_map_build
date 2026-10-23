#!/usr/bin/env python3
"""Accumulate raw M360 endpoints using encoder integration only."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from map_build.compare_odometry_to_cartographer import load_encoder, integrate_encoder, axis_to_lidar, load_cartographer, interpolate, align_first

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--windows',type=int,nargs='+',default=[1,5,20,100,500]); p.add_argument('--center-index',type=int); p.add_argument('--start-unix',type=float); p.add_argument('--end-unix',type=float); p.add_argument('--max-range',type=float,default=200); p.add_argument('--wheelbase',type=float); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    run=a.run.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    par=json.loads((run/'run_parameters.json').read_text()); stats=json.loads((run/'map/run_stats.json').read_text())
    enc=load_encoder(run/'input/encoder.jsonl'); lidar=par['lidar']; pos=lidar['position_m']; ori=lidar.get('mapping_orientation_deg',lidar['orientation_deg'])
    axis,_=integrate_encoder(enc,float(a.wheelbase or stats.get('wheelbase_m',par['robot']['wheelbase_m'])),float(stats.get('steering_offset_deg',0)),float(stats.get('encoder_speed_sign',par['lidar'].get('encoder_speed_sign',1))))
    odom=axis_to_lidar(axis,float(pos['x']),float(pos['y']),float(ori.get('yaw',0)))
    cart=load_cartographer(run/'map/trajectory.json'); odom=align_first(cart[0:1,1:4],interpolate(cart[:,0],enc[:,0],odom))
    paths=sorted((run/'input'/'horizon').glob('*.npz')); stamps=np.array([float(np.load(q)['stamp']) for q in paths]); keep=np.ones(len(paths),dtype=bool); keep &= (a.start_unix is None) | (stamps>=a.start_unix); keep &= (a.end_unix is None) | (stamps<=a.end_unix); paths=[q for q,k in zip(paths,keep) if k]; stamps=stamps[keep]; center=a.center_index if a.center_index is not None else len(paths)//2; center=max(0,min(center,len(paths)-1)); chunks=[]
    for q,stamp in zip(paths,stamps):
        with np.load(q) as d: xy=np.asarray(d['xy'],float)
        rr=np.linalg.norm(xy,axis=1); xy=xy[np.isfinite(rr)&(rr<=a.max_range)]; pose=np.array([np.interp(stamp,cart[:,0],odom[:,j]) for j in range(3)]); c,s=np.cos(pose[2]),np.sin(pose[2]); chunks.append(np.c_[pose[0]+c*xy[:,0]-s*xy[:,1],pose[1]+s*xy[:,0]+c*xy[:,1]])
    fig,axes=plt.subplots(1,len(a.windows),figsize=(6*len(a.windows),6),squeeze=False)
    for ax,w in zip(axes[0],a.windows):
        half=w//2; lo=max(0,center-half); hi=min(len(chunks),lo+w); pts=np.vstack(chunks[lo:hi]); ax.scatter(pts[:,0],pts[:,1],s=.35,c=np.arange(len(pts)),cmap='turbo',linewidths=0,rasterized=True); ax.plot(odom[:,0],odom[:,1],color='black',lw=.5,alpha=.5); cp=np.array([np.interp(stamps[center],cart[:,0],odom[:,j]) for j in range(3)]); ax.plot(cp[0],cp[1],'r+'); ax.set_aspect('equal'); ax.set_title(f'{hi-lo} scans\nframes {lo}..{hi-1}'); ax.grid(alpha=.2)
    fig.suptitle(f'Encoder-only accumulation; center scan {center}'); fig.tight_layout(); fig.savefig(out/'encoder_accumulation_windows.png',dpi=200); plt.close(fig)
    (out/'report.json').write_text(json.dumps({'center_index':int(center),'windows':a.windows,'max_range_m':a.max_range,'scan_count':len(paths)},indent=2)+'\n'); print(out/'encoder_accumulation_windows.png')
if __name__=='__main__': main()
