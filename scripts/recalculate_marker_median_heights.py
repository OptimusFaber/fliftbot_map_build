#!/usr/bin/env python3
import csv, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/home/optimus/ros-projects/fliftbot_map_build'); sys.path.insert(0,str(ROOT/'src/fliftbot_mapping/evaluation'))
import ceiling_marker_localization_stored_ray_corrected as loc
import ceiling_marker_localization_track_candidate as base

def main(marker_dir):
    marker_dir=Path(marker_dir); run=marker_dir.parent
    obs=Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/observations_camera_backup_oldpose/observations.json')
    cam=Path('/home/optimus/ros-projects/camera_backup.json'); camera_pos=Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/camera_position_lowered_5cm.txt')
    rows,_=loc.rows_from_stored(obs,cam); tt,traj,_=base.load_trajectory(run/'map/trajectory.json'); poses,inside=base.interpolate(np.array([r['timestamp_s'] for r in rows]),tt,traj)
    ld=json.loads((run/'input/lidar.markers.config').read_text()); lp=np.array([ld['position_m'][k] for k in ('x','y','z')]); cp=np.array(loc.sidecar(camera_pos)['position_m']); lever=cp[:2]-lp[:2]
    oh=poses[:,2]-math.radians(float(ld.get('orientation_deg',{}).get('yaw',180.0))); rh=oh+math.radians(float(ld['ray_correction_deg']['yaw']))
    origins=np.column_stack((poses[:,0]+np.cos(oh)*lever[0]-np.sin(oh)*lever[1],poses[:,1]+np.sin(oh)*lever[0]+np.cos(oh)*lever[1],np.full(len(poses),cp[2])))
    body=np.array([r['body_ray_unit'] for r in rows]); directions=np.column_stack((np.cos(rh)*body[:,0]-np.sin(rh)*body[:,1],np.sin(rh)*body[:,0]+np.cos(rh)*body[:,1],body[:,2]))
    segments=base.split_segments(rows); data=json.loads((marker_dir/'marker_points_stored_ray_corrected.json').read_text()); result=[]
    for m in data['markers']:
        xy=np.array(m['point_xyz_m'][:2]); idx=[i for sid,vals in enumerate(segments) if sid in set(m['track_segments']) for i in vals if inside[i]]; hs=[]
        for i in idx:
            d=directions[i,:2]; t=float(np.dot(xy-origins[i,:2],d)/np.dot(d,d)); hs.append(float(origins[i,2]+t*directions[i,2]))
        result.append({'marker_id':m['marker_id'],'point_xy_m':m['point_xyz_m'][:2],'observations':len(hs),'height_median':float(np.median(hs)) if hs else None,'height_min':float(min(hs)) if hs else None,'height_max':float(max(hs)) if hs else None,'height_spread':float(max(hs)-min(hs)) if hs else None})
    out=marker_dir/'marker_median_heights.json'; out.write_text(json.dumps({'method':'median height per marker','markers':result},indent=2)+'\n')
    with (marker_dir/'marker_median_heights.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=result[0].keys()); writer.writeheader(); writer.writerows(result)
    meta={};
    for line in (run/'map/map.yaml').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            try: meta[k.strip()]=json.loads(v.strip())
            except: meta[k.strip()]=v.strip()
    im=Image.open(run/'map'/meta['image']).convert('RGB'); dr=ImageDraw.Draw(im); res=float(meta['resolution']); ox,oy=meta['origin'][:2]
    for m in result:
        if m['height_median'] is None: continue
        x=(m['point_xy_m'][0]-ox)/res; y=im.height-(m['point_xy_m'][1]-oy)/res; dr.ellipse((x-6,y-6,x+6,y+6),fill='red',outline='white',width=2); dr.text((x+8,y-8),f"{m['marker_id'].replace('marker_','')} H={m['height_median']:.3f}",fill='black',stroke_width=2,stroke_fill='white')
    im.save(marker_dir/'map_marker_median_heights.png')
    print(json.dumps({'markers':len(result),'output':str(out)},indent=2))
if __name__=='__main__': main(sys.argv[1])
