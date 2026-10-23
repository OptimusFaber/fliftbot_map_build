#!/usr/bin/env python3
"""Highlight accumulated endpoints adjacent to a large within-scan range jump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OFFSET = 62135596800.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--jump-m", type=float, default=2.0)
    p.add_argument("--max-bearing-gap-deg", type=float, default=0.25)
    p.add_argument("--max-range", type=float, default=18.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--xlim", type=float, nargs=2)
    p.add_argument("--ylim", type=float, nargs=2)
    a = p.parse_args(); run = a.run.resolve()
    q = json.loads((run / "map" / "trajectory.json").read_text())["poses"]
    pt=np.array([z["timestamp_universal"]/1e7-OFFSET for z in q]); px=np.array([z["x"] for z in q]); py=np.array([z["y"] for z in q]); pa=np.unwrap([2*np.arctan2(z["qz"],z["qw"]) for z in q])
    regular=[]; edges=[]
    for path in sorted((run/"input/horizon").glob("*.npz"))[::a.stride]:
        with np.load(path) as z: xy=np.asarray(z["xy"],float); stamp=float(z["stamp"])
        r=np.linalg.norm(xy,axis=1); bearing=np.unwrap(np.arctan2(xy[:,1],xy[:,0]))
        order=np.argsort(bearing); sr=r[order]; sb=bearing[order]
        jump_prev=np.r_[False,(np.abs(np.diff(sr))>=a.jump_m)&(np.diff(sb)<=np.deg2rad(a.max_bearing_gap_deg))]
        jump_next=np.r_[(np.abs(np.diff(sr))>=a.jump_m)&(np.diff(sb)<=np.deg2rad(a.max_bearing_gap_deg)),False]
        edge_sorted=jump_prev|jump_next; edge=np.zeros(len(r),dtype=bool);edge[order]=edge_sorted
        valid=np.isfinite(r)&(r<=a.max_range); edge&=valid
        x=np.interp(stamp,pt,px);y=np.interp(stamp,pt,py);yaw=np.interp(stamp,pt,pa);c,s=np.cos(yaw),np.sin(yaw)
        w=np.column_stack((x+c*xy[:,0]-s*xy[:,1],y+s*xy[:,0]+c*xy[:,1]))
        regular.append(w[valid&~edge]);edges.append(w[edge])
    regular=np.vstack(regular);edges=np.vstack(edges)
    fig,ax=plt.subplots(figsize=(16,10),constrained_layout=True)
    ax.scatter(regular[:,0],regular[:,1],s=.25,c="#94a3b8",alpha=.22,linewidths=0,label="ordinary endpoint")
    ax.scatter(edges[:,0],edges[:,1],s=2,c="#e11d48",alpha=.85,linewidths=0,label=f"adjacent range jump ≥ {a.jump_m:g} m")
    ax.plot(px,py,c="black",lw=1,label="trajectory")
    ax.set(aspect="equal",xlabel="map X, m",ylabel="map Y, m",title="Endpoints next to depth discontinuities")
    if a.xlim: ax.set_xlim(a.xlim)
    if a.ylim: ax.set_ylim(a.ylim)
    ax.grid(alpha=.2);ax.legend(loc="upper right",markerscale=6)
    out=run/"diagnostics/depth_edge_endpoints";out.mkdir(parents=True,exist_ok=True);target=out/"depth_edge_endpoints.png"
    fig.savefig(target,dpi=200);plt.close(fig);print(target)


if __name__ == "__main__": main()
