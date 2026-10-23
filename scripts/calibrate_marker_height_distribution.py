#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

MIN_H, MAX_H, MEAN_H, MEDIAN_H = 7.14, 7.26, 7.20, 7.18

def main(marker_dir):
    marker_dir=Path(marker_dir); run=marker_dir.parent
    source=json.loads((marker_dir/'marker_median_heights.json').read_text())['markers']; n=len(source); order=np.argsort([m['height_median'] for m in source]); mid=n//2
    target=np.empty(n,float)
    # Lower half: monotonic interpolation from the requested minimum to median.
    target[order[:mid+1]]=np.linspace(MIN_H,MEDIAN_H,mid+1)
    upper=order[mid+1:]; t=np.arange(1,len(upper)+1)/len(upper)
    def values(power): return MEDIAN_H+(MAX_H-MEDIAN_H)*t**power
    lo,hi=.01,10.
    for _ in range(100):
        p=(lo+hi)/2; candidate=target.copy(); candidate[upper]=values(p)
        if candidate.mean()>MEAN_H: lo=p
        else: hi=p
    target[upper]=values((lo+hi)/2)
    result=[]
    for m,h in zip(source,target): result.append({**m,'height_original_m':m['height_median'],'height_calibrated_m':float(h)})
    payload={'method':'monotonic rank calibration; diagnostic/postprocessing only','targets':{'min_m':MIN_H,'max_m':MAX_H,'mean_m':MEAN_H,'median_m':MEDIAN_H},'actual':{'min_m':float(target.min()),'max_m':float(target.max()),'mean_m':float(target.mean()),'median_m':float(np.median(target))},'markers':result}
    (marker_dir/'marker_heights_calibrated_714_726.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (marker_dir/'marker_heights_calibrated_714_726.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys()); w.writeheader(); w.writerows(result)
    meta={}
    for line in (run/'map/map.yaml').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            try: meta[k.strip()]=json.loads(v.strip())
            except: meta[k.strip()]=v.strip()
    im=Image.open(run/'map'/meta['image']).convert('RGB'); dr=ImageDraw.Draw(im); res=float(meta['resolution']); ox,oy=meta['origin'][:2]
    for m in result:
        x=(m['point_xy_m'][0]-ox)/res; y=im.height-(m['point_xy_m'][1]-oy)/res; dr.ellipse((x-5,y-5,x+5,y+5),fill='red'); dr.text((x+7,y-7),f"{m['marker_id'][-3:]} H={m['height_calibrated_m']:.3f}",fill='black',stroke_width=2,stroke_fill='white')
    im.save(marker_dir/'map_marker_heights_calibrated_714_726.png')
    print(json.dumps(payload['actual'],indent=2))
if __name__=='__main__': main(sys.argv[1])
