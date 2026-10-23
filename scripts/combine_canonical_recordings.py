#!/usr/bin/env python3
"""Combine a tail of one canonical recording with a complete second recording."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np

def scans(root: Path):
    out=[]
    for p in (root/'horizon').glob('*.npz'):
        with np.load(p) as z: out.append((float(z['stamp']), p))
    return sorted(out)

def rows(path: Path):
    return [json.loads(x) for x in (path/'encoder.jsonl').read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--first-recording', type=Path, required=True)
    ap.add_argument('--first-tail-minutes', type=float, required=True)
    ap.add_argument('--second-recording', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gap-s', type=float, default=.10)
    a=ap.parse_args(); first=a.first_recording.resolve(); second=a.second_recording.resolve(); out=a.output.resolve()
    if out.exists() and any(out.iterdir()): raise SystemExit(f'output is not empty: {out}')
    s1=scans(first); s2=scans(second)
    if not s1 or not s2: raise SystemExit('both recordings must contain horizon/*.npz')
    cutoff=s1[-1][0]-a.first_tail_minutes*60
    s1=[x for x in s1 if x[0]>=cutoff]
    if not s1: raise SystemExit('first tail is empty')
    shift=s1[-1][0]+a.gap_s-s2[0][0]
    (out/'horizon').mkdir(parents=True)
    entries=[]
    for i,(off,(stamp,p)) in enumerate([(0,x) for x in s1]+[(shift,x) for x in s2],1):
        with np.load(p) as z:
            data={k:z[k] for k in z.files}
        data['stamp']=np.asarray(float(data['stamp'])+off)
        if 't' in data: data['t']=np.asarray(data['t'], dtype=float)+off
        name=f'{i:08d}_{stamp+off:.6f}.npz'
        np.savez_compressed(out/'horizon'/name, **data)
        entries.append(float(stamp+off))
    merged=[]
    r1=[r for r in rows(first) if float(r['enc_unix'])>=cutoff]
    r2=rows(second)
    for r in r1: merged.append(r)
    for r in r2:
        q=dict(r); q['enc_unix']=float(q['enc_unix'])+shift; merged.append(q)
    merged.sort(key=lambda r: float(r['enc_unix']))
    (out/'encoder.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in merged))
    for name in ('camera.config','lidar.config','liftbot.config','camera_intrinsics.json'):
        src=first/name
        if src.is_file(): shutil.copy2(src,out/name)
    report={'schema':'fliftbot.canonical_hybrid.v1','first_source':str(first),'second_source':str(second),
            'first_tail_minutes':a.first_tail_minutes,'cutoff_unix':cutoff,'second_time_shift_s':shift,
            'gap_s':a.gap_s,'scan_count':len(entries),'encoder_count':len(merged),
            'start_unix':entries[0],'end_unix':entries[-1],'duration_s':entries[-1]-entries[0]}
    (out/'hybrid_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
