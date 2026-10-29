#!/usr/bin/env python3
"""Frozen corrected localization using observations item['ray'] directly."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import ceiling_marker_localization_track_candidate as base
import _ceiling_marker_geometry as geometry

WEAK_ASSOCIATION_DISTANCE_M=.35
WEAK_PROJECTION_RESIDUAL_M=.10
WEAK_PROJECTION_RESIDUAL_RATIO=1.5
WEAK_ASSOCIATION_MARGIN_M=.15

def sidecar(path):
    lines=[x.strip() for x in path.read_text().splitlines() if x.strip()]
    pos=np.array(list(map(float,lines[0].split())))/100
    ang=np.array(list(map(float,lines[1].split())))
    return {"source":str(path.resolve()),"position_m":pos.tolist(),"source_roll_pitch_rad":ang.tolist(),
            "stored_ray_generation_tilts_rad":[-ang[0],ang[1],math.pi/2],
            "stored_ray_generation_rotation":"Rz(+pi/2) @ Ry(pitch) @ Rx(-roll)",
            "runtime_policy":"rotation already embedded by CameraModel; no second mount rotation"}

def rows_from_stored(path, camera_json=None, group_by_code=False):
    doc=json.loads(path.read_text()); times,source=base.load_frame_times(doc,path); rows=[]
    matrix = distortion = None
    if camera_json is not None:
        matrix, distortion, _, _ = geometry.load_camera_intrinsics(camera_json)
    for fi,frame in enumerate(doc.get("data",[])):
        if fi>=len(times) or not isinstance(frame,dict): continue
        for tid,item in frame.items():
            if not isinstance(item,dict): continue
            try: ray=np.asarray(item["ray"],float); code=int(item.get("code",-1))
            except (KeyError,TypeError,ValueError): continue
            if ray.shape!=(3,) or not np.isfinite(ray).all() or np.linalg.norm(ray)<1e-12: continue
            ray/=np.linalg.norm(ray)
            weight=1.0
            center=item.get("center")
            if matrix is not None and isinstance(center,list) and len(center)==2:
                optical=geometry.optical_rays(np.asarray(center,float).reshape(1,2),matrix,distortion)[0]
                weight=max(0.1,float(optical[2])**4)
            rows.append({"frame_index":fi,"timestamp_s":float(times[fi]),"track_id":(f"code:{code}" if group_by_code and code >= 0 else str(tid)),
                         "code":code,"center":center,"body_ray_unit":ray.tolist(),"screen_weight":weight})
    return rows,source

def merge_markers_with_segment_gate(tracks, radius_m, max_deviation_m):
    """Merge tracks, but abstain when repeated passes do not agree.

    The ordinary connected-component merge can hide two pass-dependent lobes:
    A is close to B, B is close to C, while A and C are not mutually
    consistent.  Keep the production merge result only when every contributing
    segment center is within ``max_deviation_m`` of their robust median.
    """
    markers = base.merge_markers(tracks, radius_m)
    if max_deviation_m is None:
        return markers, []
    by_segment = {int(t["segment_id"]): t for t in tracks}
    accepted, rejected = [], []
    for marker in markers:
        members = [by_segment[int(s)] for s in marker["track_segments"]]
        points = np.asarray([m["point_xy_m"] for m in members], dtype=float)
        robust_center = np.median(points, axis=0)
        deviations = np.linalg.norm(points - robust_center, axis=1)
        maximum = float(np.max(deviations)) if len(deviations) else 0.0
        marker["segment_center_max_deviation_m"] = maximum
        if maximum <= max_deviation_m:
            accepted.append(marker)
        else:
            rejected.append({
                "track_segments": marker["track_segments"],
                "candidate_codes": marker.get("candidate_codes", []),
                "segment_center_max_deviation_m": maximum,
                "threshold_m": max_deviation_m,
            })
    return accepted, rejected

def merge_variable_height_markers(tracks, radius_m, linkage="complete", height_consensus_m=None):
    """Merge repeated track fragments by map XY, then average their XYZ.

    Independent ray fits of the same physical marker can disagree noticeably
    in height while landing almost on top of each other in the map. Using 3D
    distance would therefore preserve obvious duplicates.
    """
    # Complete-link clustering: every pair of track centers in one group must
    # be within radius_m.  Connected components are wrong here because a chain
    # A--B--C can merge A and C even when they are more than radius_m apart.
    if linkage == "connected":
        # Merge transitive fragments when each graph edge is within the
        # requested radius. This is useful when one physical marker has
        # several pass-dependent lobes linked by intermediate observations.
        pending=set(range(len(tracks))); groups=[]
        while pending:
            seed=min(pending); pending.remove(seed); stack=[seed]; group=[seed]
            while stack:
                i=stack.pop(); pi=np.asarray(tracks[i]["point_xy_m"])
                for j in sorted(pending):
                    if np.linalg.norm(pi-np.asarray(tracks[j]["point_xy_m"]))<=radius_m:
                        pending.remove(j); stack.append(j); group.append(j)
            groups.append(group)
    else:
        pending=set(range(len(tracks))); groups=[]
        while pending:
            seed=min(pending); pending.remove(seed); group=[seed]
            changed=True
            while changed:
                changed=False
                for j in sorted(pending):
                    pj=np.asarray(tracks[j]["point_xy_m"])
                    if all(np.linalg.norm(pj-np.asarray(tracks[i]["point_xy_m"]))<=radius_m
                           for i in group):
                        pending.remove(j); group.append(j); changed=True
            groups.append(group)
    result=[]
    for group in groups:
        weights=np.asarray([tracks[i]["observations"]/max(tracks[i]["spread_m"],.01) for i in group])
        xyz=np.asarray([[*tracks[i]["point_xy_m"],tracks[i]["ceiling_z_m"]] for i in group])
        point=np.average(xyz,axis=0,weights=weights)
        gated=any("height_eligible" in tracks[i] for i in group)
        keep=[j for j,i in enumerate(group) if tracks[i].get("height_eligible",True)]
        conflict=[]
        if height_consensus_m is not None and len(keep)>=2:
            heights=xyz[keep,2]
            center_height=float(np.median(heights))
            agreeing=[j for j in keep if abs(xyz[j,2]-center_height)<=height_consensus_m]
            if len(agreeing)>=2 or (len(keep)>=3 and len(agreeing)>len(keep)//2):
                conflict=[j for j in keep if j not in agreeing]
                keep=agreeing
            elif float(np.max(heights)-np.min(heights))>height_consensus_m:
                conflict=list(keep);keep=[]
        height=float(np.average(xyz[keep,2],weights=weights[keep])) if keep else (None if gated or conflict else float(point[2]))
        result.append({"point_xyz_m":[float(point[0]),float(point[1]),height],"ceiling_z_m":height,
          "height_status":"estimated" if height is not None else ("conflicting_pass_heights" if conflict else "unobservable_stationary_tracks"),
          "height_segments":[tracks[group[j]]["segment_id"] for j in keep] if gated else [tracks[i]["segment_id"] for i in group],
          "height_excluded_segments":[tracks[i]["segment_id"] for i in group if (gated and not tracks[i].get("height_eligible",True)) or i in conflict],
          "track_segments":[tracks[i]["segment_id"] for i in group],
          "candidate_codes":sorted({code for i in group for code in tracks[i]["candidate_codes"] if code>=0}),
          "observations":int(sum(tracks[i]["observations"] for i in group)),
          "spread_m":float(np.median([tracks[i]["spread_m"] for i in group]))})
    return result

def height_uncertainty_from_parallax(origins, directions, indices, weights, fit):
    """Height change equivalent to the RMS ray-intersection scatter.

    A low scatter alone does not make a short track's fitted height reliable:
    its rays may move almost together when the assumed height changes.
    """
    rays=directions[indices]
    slopes=rays[:,:2]/rays[:,2,None]
    xy=origins[indices,:2]+(fit["height_m"]-origins[indices,2,None])*slopes
    scatter=np.linalg.norm(xy-np.asarray(fit["center_xy_m"]),axis=1)
    rms=math.sqrt(float(np.average(scatter*scatter,weights=weights)))
    return rms/math.sqrt(fit["conditioning"])

def block_height_instability(origins, directions, indices, weights, fit,
                             block_distance_m=.20):
    """Refit after omitting spatially distinct quarters of a track.

    Consecutive video frames are correlated, so the number of frames cannot
    be used as the number of independent height measurements.
    """
    blocks=[]; current=[]; anchor=None
    for index in indices:
        position=origins[index,:2]
        if anchor is None or np.linalg.norm(position-anchor)>=block_distance_m:
            if current: blocks.append(current)
            current=[]; anchor=position
        current.append(int(index))
    if current: blocks.append(current)
    if len(blocks)<6:
        return float("inf"),len(blocks),[]
    estimates=[]
    for omitted in np.array_split(np.arange(len(blocks)),4):
        keep=np.concatenate([np.asarray(blocks[i],int) for i in range(len(blocks)) if i not in omitted])
        if len(keep)<8: continue
        # Give each spatial block one vote regardless of video frame rate.
        block_weights=np.concatenate([np.full(len(blocks[i]),1/len(blocks[i]))
                                      for i in range(len(blocks)) if i not in omitted])
        estimate=base.estimate_track_height(origins,directions,keep,block_weights)
        if estimate is not None and np.isfinite(estimate["height_m"]):
            estimates.append(float(estimate["height_m"]))
    if len(estimates)<4:
        return float("inf"),len(blocks),estimates
    return float(max(abs(np.asarray(estimates)-fit["height_m"]))),len(blocks),estimates

def estimate_track_height_median(origins, directions, indices, weights,
                                 min_height_m, max_height_m):
    """Fit track height by minimizing median radial XY scatter over height."""
    o,d=origins[indices],directions[indices]
    q=d[:,:2]/d[:,2,None]
    a=o[:,:2]-o[:,2,None]*q
    w=np.clip(np.asarray(weights,float),1.0e-3,None)
    def evaluate(height):
        points=a+height*q
        center=np.median(points,axis=0)
        residual=np.linalg.norm(points-center,axis=1)
        return float(np.median(residual)),center,residual
    grid=np.linspace(min_height_m,max_height_m,121)
    values=np.asarray([evaluate(h)[0] for h in grid])
    best=int(np.argmin(values))
    lo=grid[max(0,best-1)];hi=grid[min(len(grid)-1,best+1)]
    if hi>lo:
        opt=minimize_scalar(lambda h:evaluate(h)[0],bounds=(lo,hi),method="bounded",
                            options={"xatol":1.0e-4})
        height=float(opt.x)
    else:
        height=float(grid[best])
    median_residual,center,residual=evaluate(height)
    q_mean=np.average(q,axis=0,weights=w)
    conditioning=float(np.sum(w[:,None]*(q-q_mean)**2))
    return {"height_m":height,"center_xy_m":center.tolist(),
            "median_residual_m":median_residual,
            "p95_residual_m":float(np.percentile(residual,95)),
            "conditioning":conditioning,
            "height_fit_objective":"median_radial_xy_scatter"}

def merge_with_weak_height_association(tracks, origins, directions, camera_z,
                                       radius_m, linkage, max_uncertainty_m, height_consensus_m=None,
                                       max_association_distance_m=WEAK_ASSOCIATION_DISTANCE_M,
                                       max_projection_residual_m=WEAK_PROJECTION_RESIDUAL_M):
    """Attach weak-parallax tracks using rays at a nearby reliable height.

    No decoder code or GT coordinate is used. Ambiguous or unsupported tracks
    remain visible as candidates with an unobservable height.
    """
    strong=[t for t in tracks if t["height_reliable"]]
    weak=[t for t in tracks if not t["height_reliable"]]
    markers=merge_variable_height_markers(strong,radius_m,linkage,height_consensus_m) if strong else []
    weights={i:sum(t["observations"]/max(t["spread_m"],.01)
                   for t in strong if t["segment_id"] in m["track_segments"])
             for i,m in enumerate(markers)}
    associations=[];unresolved=[]
    for track in weak:
        idx=track["indices"]
        slopes=directions[idx,:2]/directions[idx,2,None]
        candidates=[]
        for mi,marker in enumerate(markers):
            height=marker["ceiling_z_m"]
            if height is None:continue
            points=origins[idx,:2]+(height-camera_z)*slopes
            center=np.median(points,axis=0)
            residual=float(np.median(np.linalg.norm(points-center,axis=1)))
            distance=float(np.linalg.norm(center-np.asarray(marker["point_xyz_m"][:2])))
            residual_limit=min(max_projection_residual_m,
                               max(.03,WEAK_PROJECTION_RESIDUAL_RATIO*track["spread_m"]))
            if distance<=min(radius_m,max_association_distance_m) and residual<=residual_limit:
                candidates.append((distance,residual,mi,center))
        candidates.sort(key=lambda c:c[0])
        # Close competing anchors mean that the physical association is not
        # established. Keep the track rather than guessing its marker.
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<WEAK_ASSOCIATION_MARGIN_M):
            unresolved.append(track)
            continue
        distance,residual,mi,center=candidates[0]
        marker=markers[mi]
        weight=track["observations"]/max(residual,.01)
        old_weight=weights[mi]
        old_xy=np.asarray(marker["point_xyz_m"][:2])
        new_xy=(old_weight*old_xy+weight*center)/(old_weight+weight)
        marker["point_xyz_m"][:2]=new_xy.tolist()
        weights[mi]+=weight
        marker["track_segments"].append(track["segment_id"])
        marker["height_excluded_segments"].append(track["segment_id"])
        marker["candidate_codes"]=sorted(set(marker["candidate_codes"])|set(track["candidate_codes"]))
        marker["observations"]+=track["observations"]
        detail={"segment_id":track["segment_id"],"fitted_height_m":track["ceiling_z_m"],
                "height_uncertainty_m":track["height_uncertainty_m"],
                "block_height_instability_m":track["block_height_instability_m"],
                "independent_spatial_blocks":track["independent_spatial_blocks"],
                "distance_at_marker_height_m":distance,"projection_residual_m":residual}
        marker.setdefault("weak_height_associations",[]).append(detail)
        associations.append(detail)
    if unresolved:
        uncertain=merge_variable_height_markers(unresolved,radius_m,linkage)
        for marker in uncertain:
            marker["point_xyz_m"][2]=None
            marker["ceiling_z_m"]=None
            marker["height_status"]="unobservable_weak_parallax"
            marker["height_segments"]=[]
            marker["height_excluded_segments"]=list(marker["track_segments"])
        markers.extend(uncertain)
    return markers,associations,len(unresolved)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("observations","trajectory","map-dir","camera-position","camera-json","lidar-config","out-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--min-segment-observations",type=int,default=8)
    p.add_argument("--min-camera-baseline-m",type=float,default=.35)
    p.add_argument("--max-track-height-uncertainty-m",type=float,default=.25,
                   help="Maximum ray-scatter/parallax height sensitivity for an independent track height")
    p.add_argument("--height-quality-method",choices=("legacy","block_cv"),default="legacy")
    p.add_argument("--block-distance-m",type=float,default=.20)
    p.add_argument("--max-block-height-instability-m",type=float,default=.30)
    p.add_argument("--cross-pass-height-consensus-m",type=float,default=None)
    p.add_argument("--track-height-fit-method",choices=("least_squares","median"),default="least_squares",
                   help="Per-track height objective; median minimizes median radial XY ray scatter")
    p.add_argument("--max-height-sigma-m",type=float,default=.12)
    p.add_argument("--min-height-confidence",type=float,default=.25)
    p.add_argument("--max-marker-height-m",type=float,default=None,
                   help="Reject independently fitted marker tracks above this absolute height")
    p.add_argument("--merge-radius-m",type=float,default=.10)
    p.add_argument("--merge-linkage",choices=("complete","connected"),default="complete")
    p.add_argument("--time-offset-s",type=float,default=0.0)
    p.add_argument("--max-segment-center-deviation-m",type=float,default=None,
                   help="Reject a merged marker when any pass center differs from their median by more than this")
    p.add_argument("--ceiling-z",type=float,default=None,help="Use a measured ceiling plane and skip automatic height consensus")
    p.add_argument("--per-track-height",action="store_true",
                   help="Estimate an independent XYZ position for every persistent marker track")
    p.add_argument("--stationary-height-filter",action="store_true",
                   help="Preserve XY from all tracks, but exclude a segment from height fusion if one near-stationary position group contains over half its observations or fewer than 20 groups remain")
    p.add_argument("--stationary-group-radius-m",type=float,default=.02)
    p.add_argument("--stationary-max-group-fraction",type=float,default=.5)
    p.add_argument("--stationary-min-groups",type=int,default=20)
    p.add_argument("--group-by-code",action="store_true",help="Group detector-only observations by decoded code when no persistent tracker IDs exist")
    a=p.parse_args(); camera_meta=sidecar(a.camera_position)
    CAMERA=np.asarray(camera_meta["position_m"], dtype=float)
    ld=json.loads(a.lidar_config.read_text()); lidar=np.array([ld["position_m"][k] for k in ("x","y","z")])
    LEVER=CAMERA[:2]-lidar[:2]
    correction=ld.get("ray_correction_deg", {})
    if not all(k in correction for k in ("roll","pitch","yaw")):
        raise ValueError("lidar.config must contain ray_correction_deg roll/pitch/yaw")
    # The physical DePHAN installation faces backward in this recording.
    # Keep the small map correction from lidar.config and include the 180 deg
    # mounting yaw explicitly.
    mount_yaw_deg = float(ld.get("orientation_deg", {}).get("yaw", 180.0))
    MOUNT_YAW_RAD=math.radians(mount_yaw_deg)
    RAY_YAW_CORRECTION_RAD=math.radians(float(correction["yaw"]))
    print(json.dumps({"camera_position_m":CAMERA.tolist(),"camera_angles_rad":camera_meta["source_roll_pitch_rad"],"lidar_position_m":lidar.tolist(),"ray_correction_deg":correction,"time_offset_s":a.time_offset_s},ensure_ascii=False))
    rows,frame_source=rows_from_stored(a.observations,a.camera_json,a.group_by_code)
    tt,traj,traj_meta=base.load_trajectory(a.trajectory)
    poses,inside=base.interpolate(np.array([r["timestamp_s"] for r in rows])+a.time_offset_s,tt,traj)
    # The Cartographer pose is map->LiDAR.  Rotate the physical camera/LiDAR
    # lever arm by the LiDAR/robot yaw.  Stored rays have a separate frame
    # contract: CameraModel already embedded the camera mount rotation.  Stored
    # rays are therefore rotated directly by robot/body yaw; do not apply an
    # additional +/-pi/2 here.
    # trajectory.json stores the pose of the LiDAR tracking frame.  The
    # stored camera rays and the camera->LiDAR lever arm are expressed in the
    # robot/body frame, so convert the LiDAR heading back to body heading
    # before applying either one.  Applying the mount yaw only to the ray,
    # while leaving the lever arm at the LiDAR heading, mirrors the camera
    # origin by roughly 2*|camera-lidar| for a 180-degree mount.
    origin_heading=poses[:,2]-MOUNT_YAW_RAD
    oc,os=np.cos(origin_heading),np.sin(origin_heading)
    ray_heading=origin_heading+RAY_YAW_CORRECTION_RAD
    rc,rs=np.cos(ray_heading),np.sin(ray_heading); body=np.array([r["body_ray_unit"] for r in rows])
    origins=np.column_stack((poses[:,0]+oc*LEVER[0]-os*LEVER[1],poses[:,1]+os*LEVER[0]+oc*LEVER[1],np.full(len(poses),CAMERA[2])))
    directions=np.column_stack((rc*body[:,0]-rs*body[:,1],rs*body[:,0]+rc*body[:,1],body[:,2]))
    valid=inside&(directions[:,2]>1e-6)&np.isfinite(directions).all(1)
    segments=[]; recs=[]; motion_by_segment={}
    for sid,values in enumerate(base.split_segments(rows)):
        idx=np.array([i for i in values if valid[i]],int)
        if len(idx)<a.min_segment_observations: continue
        groups=[]; current=[int(idx[0])]; anchor=origins[idx[0],:2]
        for oi in idx[1:]:
            if np.linalg.norm(origins[oi,:2]-anchor)>a.stationary_group_radius_m:
                groups.append(current); current=[int(oi)]; anchor=origins[oi,:2]
            else: current.append(int(oi))
        groups.append(current)
        max_group_fraction=max(map(len,groups))/len(idx)
        motion_by_segment[sid]={"independent_position_groups":len(groups),"max_group_fraction":max_group_fraction,
            "height_eligible":not a.stationary_height_filter or (len(groups)>=a.stationary_min_groups and max_group_fraction<=a.stationary_max_group_fraction)}
        codes=Counter(rows[i]["code"] for i in idx if rows[i]["code"]>=0)
        segments.append({"segment_id":sid,"indices":idx,"observations":len(idx),
                         "codes":[k for k,_ in codes.most_common()],"code_votes":dict(codes)})
        baseline=float(np.max(np.linalg.norm(origins[idx,:2]-origins[idx[0],:2],axis=1)))
        if baseline<a.min_camera_baseline_m: continue
        fit_weights=np.asarray([rows[i].get("screen_weight",1.0) for i in idx],float)
        fit=base.estimate_track_height(origins,directions,idx,fit_weights)
        if fit is not None and a.track_height_fit_method=="median":
            fit=estimate_track_height_median(origins,directions,idx,fit_weights,
                                               CAMERA[2]+.5,CAMERA[2]+15)
        if fit is None: continue
        recs.append({"segment_id":sid,"indices":idx,"observations":len(idx),"baseline_m":baseline,
                     "codes":[k for k,_ in codes.most_common()],**fit})
    # The recording contains marker planes around 12--13 m above the map
    # origin.  The old camera-height+8 m cap silently discarded the whole
    # starting corridor, even when the ray fit residual was centimetric.
    plausible=[r for r in recs if CAMERA[2]+.5<=r["height_m"]<=CAMERA[2]+15 and r["median_residual_m"]<=.20]
    above_max_height=0
    if a.max_marker_height_m is not None:
        below_limit=[r for r in plausible if r["height_m"]<=a.max_marker_height_m]
        above_max_height=len(plausible)-len(below_limit)
        plausible=below_limit
    hs=np.array([r["height_m"] for r in plausible]); ceiling=None
    height_inliers=np.ones(len(plausible),dtype=bool)
    height_consensus_ok=False
    hrep={"candidate_segments":len(plausible),"decision":"abstain_no_consensus"}
    if a.ceiling_z is not None:
        ceiling=float(a.ceiling_z)
        height_inliers=np.abs(hs-ceiling)<=max(.12,3*a.max_height_sigma_m)
        height_consensus_ok=bool(height_inliers.any())
        hrep={"decision":"fixed_measured_plane","ceiling_z_m":ceiling,"candidate_segments":len(plausible)}
    elif len(hs)>=8:
        ctr=float(np.median(hs)); mad=float(np.median(abs(hs-ctr))); keep=abs(hs-ctr)<=max(.12,3*1.4826*mad)
        est=float(np.median(hs[keep])); sig=float(1.4826*np.median(abs(hs[keep]-est)))
        conf=min(1.,keep.sum()/20)*math.exp(-sig/.10); ok=keep.sum()>=8 and sig<=a.max_height_sigma_m and conf>=a.min_height_confidence
        hrep={"candidate_segments":len(plausible),"inlier_segments":int(keep.sum()),"ceiling_z_m":est,
              "robust_sigma_m":sig,"confidence":conf,"decision":"accepted" if ok else "abstain_unstable_common_plane"}
        if ok:
            ceiling=est
            height_inliers=keep
            height_consensus_ok=True
    tracks=[]
    if a.per_track_height:
        # In this mode markers are not assumed to lie on one ceiling plane.
        # The per-track fit, physical height bounds and residual threshold are
        # the quality gates; a population-level height consensus would reject
        # valid markers mounted at different heights.
        per_track_candidates=plausible
        for r in per_track_candidates:
            if r["median_residual_m"]<=.08:
                screen_weights=np.asarray([rows[i].get("screen_weight",1.0) for i in r["indices"]],float)
                height_uncertainty=height_uncertainty_from_parallax(
                    origins,directions,r["indices"],screen_weights,r)
                if a.height_quality_method=="block_cv":
                    block_instability,block_count,block_estimates=block_height_instability(
                        origins,directions,r["indices"],screen_weights,r,a.block_distance_m)
                else:
                    block_instability,block_count,block_estimates=None,0,[]
                height_reliable=(height_uncertainty<=a.max_track_height_uncertainty_m and
                    (block_instability is None or block_instability<=a.max_block_height_instability_m))
                tracks.append({"segment_id":r["segment_id"],"point_xy_m":r["center_xy_m"],
                  "ceiling_z_m":r["height_m"],"observations":r["observations"],
                  "spread_m":r["median_residual_m"],"candidate_codes":r["codes"],
                  "indices":r["indices"],"height_uncertainty_m":height_uncertainty,
                  "height_reliable":height_reliable,
                  "block_height_instability_m":block_instability if block_instability is not None and np.isfinite(block_instability) else None,
                  "independent_spatial_blocks":block_count,"block_height_estimates_m":block_estimates,
                  "height_eligible":motion_by_segment[r["segment_id"]]["height_eligible"],
                  "independent_position_groups":motion_by_segment[r["segment_id"]]["independent_position_groups"],
                  "max_group_fraction":motion_by_segment[r["segment_id"]]["max_group_fraction"],
                  "code_votes":next(s["code_votes"] for s in segments if s["segment_id"]==r["segment_id"])})
        heights=np.asarray([t["ceiling_z_m"] for t in tracks])
        eligible_heights=np.asarray([t["ceiling_z_m"] for t in tracks if t["height_eligible"]])
        hrep={"decision":"independent_track_heights","candidate_segments":len(plausible),
          "height_consensus_inliers":None,
          "height_outlier_segments_rejected":0,
          "accepted_segments":len(tracks),"height_min_m":float(np.min(heights)) if len(heights) else None,
          "height_median_m":float(np.median(heights)) if len(heights) else None,
          "height_max_m":float(np.max(heights)) if len(heights) else None,
          "stationary_height_filter":{"enabled":a.stationary_height_filter,"group_radius_m":a.stationary_group_radius_m,
            "max_group_fraction":a.stationary_max_group_fraction,"min_groups":a.stationary_min_groups,
            "eligible_segments":int(len(eligible_heights)),"excluded_segments":int(len(heights)-len(eligible_heights)),
            "eligible_height_min_m":float(np.min(eligible_heights)) if len(eligible_heights) else None,
            "eligible_height_median_m":float(np.median(eligible_heights)) if len(eligible_heights) else None,
            "eligible_height_max_m":float(np.max(eligible_heights)) if len(eligible_heights) else None}}
        ceiling=None
    elif ceiling is not None:
        pts=origins[:,:2]+((ceiling-origins[:,2])/directions[:,2])[:,None]*directions[:,:2]
        for r in segments:
            idx=r["indices"]; pp=pts[idx]; ctr=np.median(pp,0); dist=np.linalg.norm(pp-ctr,axis=1)
            keep=dist<=max(.08,float(np.percentile(dist,80)))
            if keep.sum()<a.min_segment_observations: continue
            ctr=np.median(pp[keep],0); spread=float(np.median(np.linalg.norm(pp[keep]-ctr,axis=1)))
            if spread<=.08: tracks.append({"segment_id":r["segment_id"],"point_xy_m":ctr.tolist(),"ceiling_z_m":ceiling,
                "observations":int(keep.sum()),"spread_m":spread,"candidate_codes":r["codes"],
                "code_votes":r["code_votes"]})
    if a.per_track_height:
        markers,weak_height_associations,unresolved_weak_tracks=(
          merge_with_weak_height_association(
            tracks,origins,directions,CAMERA[2],a.merge_radius_m,a.merge_linkage,
            a.max_track_height_uncertainty_m,a.cross_pass_height_consensus_m) if tracks else ([],[],0))
        rejected_segment_disagreement=[]
    else:
        markers,rejected_segment_disagreement=merge_markers_with_segment_gate(
            tracks,a.merge_radius_m,a.max_segment_center_deviation_m) if tracks else ([],[])
        weak_height_associations=[];unresolved_weak_tracks=0
    tracks_by_segment={int(track["segment_id"]):track for track in tracks}
    markers.sort(key=lambda marker:(marker["point_xyz_m"][0],marker["point_xyz_m"][1]))
    for index,marker in enumerate(markers,1):
        votes=Counter()
        for segment_id in marker["track_segments"]:
            votes.update({int(code):int(count) for code,count in tracks_by_segment[int(segment_id)].get("code_votes",{}).items()})
        ranked=votes.most_common()
        decoder_label=(ranked[0][0] if ranked and (len(ranked)==1 or ranked[0][1]>ranked[1][1]) else "?")
        marker["marker_id"] = f"marker_{index:03d}"
        marker["decoder_label"] = decoder_label
        marker["decoder_label_votes"] = {str(code):count for code,count in ranked}
        for detail in marker.get("weak_height_associations",[]):
            detail["marker_id"]=marker["marker_id"]
    status="accepted" if (a.per_track_height or ceiling is not None) and len(markers)>=3 else "abstained"
    if status=="abstained": markers=[]
    a.out_dir.mkdir(parents=True,exist_ok=True); png=a.out_dir/"stored_ray_corrected_map.png"
    base.render(png,a.map_dir,traj,origins[valid],directions[valid],markers,status,ceiling)
    report={"schema":"fliftbot.ceiling_marker_localization.stored_ray_corrected.v1","status":status,
      "runtime_gt_used":False,"runtime_registration_used":False,
      "inputs":{"observations":str(a.observations.resolve()),"trajectory":str(a.trajectory.resolve()),"camera_position":str(a.camera_position.resolve()),"lidar_config":str(a.lidar_config.resolve()),"frame_timestamps":frame_source},
      "frame_contract":{"camera_axis_m":CAMERA.tolist(),"lidar_axis_m":lidar.tolist(),"camera_from_lidar_xy_m":LEVER.tolist(),"origin_z_m":CAMERA[2],"origin_heading":"quaternion_yaw - lidar mounting yaw","ray_heading":"body yaw + independent ray yaw correction; stored CameraModel ray used directly","time":f"direct Unix timestamps; dt={a.time_offset_s:+.6f}s","time_offset_s":a.time_offset_s,"lidar_mount_yaw_deg":mount_yaw_deg,"lidar_mount_yaw_rad":MOUNT_YAW_RAD,"ray_yaw_correction_rad":RAY_YAW_CORRECTION_RAD,"trajectory_tracking_frame":"lidar","camera_sidecar":camera_meta,"trajectory_metadata":traj_meta},
      "geometry":{"ray_source":"observations.json item['ray'] direct","bbox_used":False,"center_used":False,"camera_model_rebuild_used":False,"primitive":"continuous track_id segments","decoder_code_role":"metadata only"},
      "height":{"track_fit_method":a.track_height_fit_method,**hrep},"quality_thresholds":{"min_camera_baseline_m":a.min_camera_baseline_m,"track_height_fit_method":a.track_height_fit_method,"max_track_height_uncertainty_m":a.max_track_height_uncertainty_m,"height_quality_method":a.height_quality_method,"block_distance_m":a.block_distance_m,"max_block_height_instability_m":a.max_block_height_instability_m,"cross_pass_height_consensus_m":a.cross_pass_height_consensus_m,"weak_association_distance_m":WEAK_ASSOCIATION_DISTANCE_M,"weak_projection_residual_m":WEAK_PROJECTION_RESIDUAL_M,"weak_projection_residual_ratio":WEAK_PROJECTION_RESIDUAL_RATIO,"weak_association_margin_m":WEAK_ASSOCIATION_MARGIN_M,"max_height_sigma_m":a.max_height_sigma_m,"min_height_confidence":a.min_height_confidence,"max_marker_height_m":a.max_marker_height_m,"max_segment_center_deviation_m":a.max_segment_center_deviation_m,"merge_radius_m":a.merge_radius_m,"stationary_height_filter":a.stationary_height_filter,"stationary_group_radius_m":a.stationary_group_radius_m,"stationary_max_group_fraction":a.stationary_max_group_fraction,"stationary_min_groups":a.stationary_min_groups},"counts":{"stored_ray_rows":len(rows),"projected_rows":int(valid.sum()),"track_segments_available":len(segments),"track_segments_fitted":len(recs),"plausible_height_segments":len(plausible),"segments_rejected_above_max_height":above_max_height,"track_markers_before_merge":len(tracks),"weak_height_tracks":sum(not t["height_reliable"] for t in tracks) if a.per_track_height else 0,"weak_height_tracks_associated":len(weak_height_associations),"weak_height_tracks_unresolved":unresolved_weak_tracks,"markers_rejected_segment_disagreement":len(rejected_segment_disagreement),"physical_markers":len(markers),"stationary_height_segments_excluded":sum(1 for t in tracks if a.stationary_height_filter and not t["height_eligible"]),"markers_without_observable_height":sum(m.get("ceiling_z_m") is None for m in markers)},"markers":markers,"weak_height_associations":weak_height_associations,"segment_disagreement_rejections":rejected_segment_disagreement,"outputs":{"map_png":str(png.resolve())},
      "forbidden_inputs":["bbox","center","CameraModel ray rebuild","GT","marker 98","floorplan","time fit","parameter sweep","SE2 fit","scale fit"]}
    report["outputs"]["map_closeup_png"] = str(png.with_name(png.stem + "_closeup" + png.suffix).resolve())
    if a.per_track_height:
        report["track_height_quality"]=[{"segment_id":t["segment_id"],
            "fitted_height_m":t["ceiling_z_m"],"observations":t["observations"],
            "height_reliable":t["height_reliable"],"height_uncertainty_m":t["height_uncertainty_m"],
            "block_height_instability_m":t["block_height_instability_m"],
            "independent_spatial_blocks":t["independent_spatial_blocks"],
            "block_height_estimates_m":t["block_height_estimates_m"]} for t in tracks]
    out=a.out_dir/"marker_points_stored_ray_corrected.json"; out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":status,"height":hrep,"markers":len(markers),"json":str(out.resolve()),"png":str(png.resolve())},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
