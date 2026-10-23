#!/usr/bin/env python3
import csv, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/optimus/ros-projects/fliftbot_map_build')
sys.path.insert(0, str(ROOT / 'src/fliftbot_mapping/evaluation'))
import ceiling_marker_localization_stored_ray_corrected as loc
import ceiling_marker_localization_track_candidate as base

def main(marker_dir: Path, marker_number=22):
    run = marker_dir.parent
    obs = Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/observations_camera_backup_oldpose/observations.json')
    cam = Path('/home/optimus/ros-projects/camera_backup.json')
    camera_pos = Path('/home/optimus/ros-projects/ros-data/results/real/20261015/sweep_5cm_12min_18m/camera_position_lowered_5cm.txt')
    lidar = run / 'input/lidar.markers.config'
    rows, _ = loc.rows_from_stored(obs, cam)
    tt, traj, _ = base.load_trajectory(run / 'map/trajectory.json')
    poses, inside = base.interpolate(np.array([r['timestamp_s'] for r in rows]), tt, traj)
    ld = json.loads(lidar.read_text()); lp = np.array([ld['position_m'][k] for k in ('x','y','z')])
    cp = np.array(loc.sidecar(camera_pos)['position_m']); lever = cp[:2] - lp[:2]
    mount = math.radians(float(ld.get('orientation_deg', {}).get('yaw', 180.0)))
    yawcorr = math.radians(float(ld['ray_correction_deg']['yaw']))
    oh = poses[:,2] - mount; rh = oh + yawcorr
    origins = np.column_stack((poses[:,0]+np.cos(oh)*lever[0]-np.sin(oh)*lever[1], poses[:,1]+np.sin(oh)*lever[0]+np.cos(oh)*lever[1], np.full(len(poses),cp[2])))
    body = np.array([r['body_ray_unit'] for r in rows]); directions = np.column_stack((np.cos(rh)*body[:,0]-np.sin(rh)*body[:,1], np.sin(rh)*body[:,0]+np.cos(rh)*body[:,1], body[:,2]))
    segments = base.split_segments(rows)
    marker_name = f'marker_{marker_number:03d}'
    target = next(m for m in json.loads((marker_dir/'marker_points_stored_ray_corrected.json').read_text())['markers'] if m['marker_id']==marker_name)
    xy = np.array(target['point_xyz_m'][:2]); wanted = set(target['track_segments']); idx = [i for sid, values in enumerate(segments) if sid in wanted for i in values if inside[i]]
    out=[]
    for i in idx:
        d=directions[i,:2]; o=origins[i,:2]; t=float(np.dot(xy-o,d)/np.dot(d,d)); z=float(origins[i,2]+t*directions[i,2]); u,v=rows[i]['center']; rn=math.hypot((u-960)/960,(v-540)/960); col='center' if rn<.33 else ('middle' if rn<.66 else 'edge')
        out.append({'index':i,'frame':rows[i]['frame_index'],'u_px':u,'v_px':v,'part':col,'weight':rows[i].get('screen_weight',1.0),'height_m':z})
    out.sort(key=lambda x:x['frame'])
    (marker_dir/f'marker_{marker_number:03d}_observation_heights.json').write_text(json.dumps({'marker':str(marker_number),'track_segments':sorted(wanted),'point_xy_m':xy.tolist(),'observations':out},indent=2)+'\n')
    with (marker_dir/f'marker_{marker_number:03d}_observation_heights.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    im=Image.new('RGB',(1920,1080),'white'); dr=ImageDraw.Draw(im)
    for x in (640,1280): dr.line((x,0,x,1080),fill='gray',width=2)
    for y in (360,720): dr.line((0,y,1920,y),fill='gray',width=2)
    for label,x,y in [('left/top',20,20),('center/top',660,20),('right/top',1300,20),('left/middle',20,380),('center/middle',660,380),('right/middle',1300,380),('left/bottom',20,740),('center/bottom',660,740),('right/bottom',1300,740)]: dr.text((x,y),label,fill='gray')
    lo=min(x['height_m'] for x in out); hi=max(x['height_m'] for x in out)
    for x in out:
        q=0 if hi==lo else (x['height_m']-lo)/(hi-lo); color=(int(255*q),30,int(255*(1-q))); r=6; u,v=x['u_px'],x['v_px']; dr.ellipse((u-r,v-r,u+r,v+r),fill=color,outline='black'); dr.text((u+8,v-8),f"H={x['height_m']:.3f} W={x['weight']:.3f}",fill='black',stroke_width=1,stroke_fill='white')
    dr.rectangle((10,1030,600,1070),fill='white',outline='black')
    dr.text((20,1040),f"marker {marker_number:03d} | n={len(out)} | H: {lo:.3f}..{hi:.3f} m | W=cos⁴(theta), min 0.1",fill='black')
    im.save(marker_dir/f'marker_{marker_number:03d}_observation_heights_screen.png')
    print(json.dumps({'n':len(out),'min_m':lo,'median_m':float(np.median([x['height_m'] for x in out])),'max_m':hi,'spread_m':hi-lo},indent=2))

if __name__=='__main__': main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 22)
