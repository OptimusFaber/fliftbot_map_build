#!/usr/bin/env python3
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/home/optimus/ros-projects/fliftbot_map_build'); sys.path.insert(0,str(ROOT/'src/fliftbot_mapping/evaluation'))
import ceiling_marker_localization_stored_ray_corrected as loc
import ceiling_marker_localization_track_candidate as base

def solve(origins,directions,weights):
    eye=np.eye(3); A=np.zeros((3,3)); b=np.zeros(3)
    for o,d,w in zip(origins,directions,weights):
        m=eye-np.outer(d,d); A+=w*m; b+=w*m@o
    if np.linalg.cond(A)>1e10: return None
    return np.linalg.solve(A,b)

def fit(origins,directions,screen_weights,robust=False):
    w=np.array(screen_weights,float); point=solve(origins,directions,w)
    for _ in range(8 if robust else 0):
        residual=np.linalg.norm(np.cross(directions,point-origins),axis=1); med=np.median(residual); scale=max(1.4826*np.median(abs(residual-med)),.01); huber=np.minimum(1.,(1.5*scale)/np.maximum(residual,1e-9)); point=solve(origins,directions,w*huber)
    residual=np.linalg.norm(np.cross(directions,point-origins),axis=1)
    return point,residual

def main(marker_dir):
    marker_dir=Path(marker_dir); run=marker_dir.parent
    obs=Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/observations_camera_backup_oldpose/observations.json'); cam=Path('/home/optimus/ros-projects/camera_backup.json'); camera_pos=Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/camera_position_lowered_5cm.txt')
    rows,_=loc.rows_from_stored(obs,cam); tt,traj,_=base.load_trajectory(run/'map/trajectory.json'); poses,inside=base.interpolate(np.array([r['timestamp_s'] for r in rows]),tt,traj)
    ld=json.loads((run/'input/lidar.markers.config').read_text()); lp=np.array([ld['position_m'][k] for k in ('x','y','z')]); cp=np.array(loc.sidecar(camera_pos)['position_m']); lever=cp[:2]-lp[:2]
    oh=poses[:,2]-math.radians(float(ld.get('orientation_deg',{}).get('yaw',180))); rh=oh+math.radians(float(ld['ray_correction_deg']['yaw'])); body=np.array([r['body_ray_unit'] for r in rows])
    origins=np.column_stack((poses[:,0]+np.cos(oh)*lever[0]-np.sin(oh)*lever[1],poses[:,1]+np.sin(oh)*lever[0]+np.cos(oh)*lever[1],np.full(len(poses),cp[2]))); directions=np.column_stack((np.cos(rh)*body[:,0]-np.sin(rh)*body[:,1],np.sin(rh)*body[:,0]+np.cos(rh)*body[:,1],body[:,2]))
    segments=base.split_segments(rows); markers=json.loads((marker_dir/'marker_points_stored_ray_corrected.json').read_text())['markers']; variants={'unweighted':(False,False),'cos4':(True,False),'cos4_robust':(True,True)}
    all_results={}
    for name,(weighted,robust) in variants.items():
        result=[]
        for m in markers:
            ids=set(m['track_segments']); idx=np.array([i for sid,vals in enumerate(segments) if sid in ids for i in vals if inside[i]],int)
            sw=np.array([rows[i].get('screen_weight',1.) for i in idx]) if weighted else np.ones(len(idx)); fitted=fit(origins[idx],directions[idx],sw,robust)
            if fitted is None: continue
            p,res=fitted; result.append({'marker_id':m['marker_id'],'point_xyz_m':p.tolist(),'observations':len(idx),'residual_median':float(np.median(res)),'residual_p95':float(np.percentile(res,95))})
        all_results[name]=result
        (marker_dir/f'ray_intersection_3d_{name}.json').write_text(json.dumps({'method':name,'markers':result},indent=2)+'\n')
    meta={}
    for line in (run/'map/map.yaml').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            try: meta[k.strip()]=json.loads(v.strip())
            except: meta[k.strip()]=v.strip()
    for name,result in all_results.items():
        im=Image.open(run/'map'/meta['image']).convert('RGB'); dr=ImageDraw.Draw(im); res=float(meta['resolution']); ox,oy=meta['origin'][:2]
        for m in result:
            x=(m['point_xyz_m'][0]-ox)/res; y=im.height-(m['point_xyz_m'][1]-oy)/res; dr.ellipse((x-5,y-5,x+5,y+5),fill='red'); dr.text((x+7,y-7),f"{m['marker_id'][-3:]} H={m['point_xyz_m'][2]:.3f}",fill='black',stroke_width=2,stroke_fill='white')
        im.save(marker_dir/f'ray_intersection_3d_{name}.png')
    summary={name:{'markers':len(v),'height_median':float(np.median([m['point_xyz_m'][2] for m in v])),'residual_median':float(np.median([m['residual_median'] for m in v]))} for name,v in all_results.items()}
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main(sys.argv[1])
