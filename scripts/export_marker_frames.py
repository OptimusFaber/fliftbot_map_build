#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, '/home/optimus/ros-projects/fliftbot_map_build/src/fliftbot_mapping/evaluation')
import ceiling_marker_localization_stored_ray_corrected as loc
import ceiling_marker_localization_track_candidate as base

def main(marker_dir):
    marker_dir=Path(marker_dir); run=marker_dir.parent
    obs_path=Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/observations_camera_backup_oldpose/observations.json')
    doc=json.loads(obs_path.read_text()); rows,_=loc.rows_from_stored(obs_path,Path('/home/optimus/ros-projects/camera_backup.json'))
    target=next(m for m in json.loads((marker_dir/'marker_points_stored_ray_corrected.json').read_text())['markers'] if m['marker_id']=='marker_017')
    wanted=set(target['track_segments']); segments=base.split_segments(rows); selected=[]
    for sid,values in enumerate(segments):
        if sid in wanted:
            selected.extend(values)
    by_frame={}
    for i in selected:
        r=rows[i]; frame=doc['data'][r['frame_index']]; item=frame.get(r['track_id'])
        if isinstance(item,dict) and len(item.get('tlbr',[]))==4:
            by_frame.setdefault(r['frame_index'],[]).append((item,r['timestamp_s'],r['track_id']))
    video=doc['video_path']; cap=cv2.VideoCapture(video); out_dir=marker_dir/'marker_017_frames_with_boxes'; out_dir.mkdir(exist_ok=True)
    index=[]
    wanted_frames=sorted(by_frame); wanted_pos=0; frame_idx=0
    while wanted_pos < len(wanted_frames):
        ok,img=cap.read()
        if not ok: continue
        if frame_idx != wanted_frames[wanted_pos]: frame_idx += 1; continue
        items=by_frame[frame_idx]
        for item,ts,tid in items:
            x1,y1,x2,y2=[int(round(v)) for v in item['tlbr']]; cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4); cv2.putText(img,f'marker 017  frame={frame_idx}',(x1,max(35,y1-12)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        name=f'frame_{frame_idx:06d}.png'; cv2.imwrite(str(out_dir/name),img); index.append({'frame':frame_idx,'timestamp_s':items[0][1],'file':name,'boxes':[x[0]['tlbr'] for x in items]}); wanted_pos += 1; frame_idx += 1
    cap.release()
    with (marker_dir/'marker_017_frames_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['frame','timestamp_s','file','boxes']); w.writeheader();
        for x in index: w.writerow({'frame':x['frame'],'timestamp_s':x['timestamp_s'],'file':x['file'],'boxes':x['boxes']})
    (marker_dir/'marker_017_frames_index.json').write_text(json.dumps({'marker':'017','track_segments':sorted(wanted),'frames':index},indent=2)+'\n')
    print(json.dumps({'frames_saved':len(index),'directory':str(out_dir)},indent=2))
if __name__=='__main__': main(sys.argv[1])
