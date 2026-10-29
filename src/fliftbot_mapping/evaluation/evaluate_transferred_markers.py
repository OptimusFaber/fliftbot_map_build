#!/usr/bin/env python3
"""Evaluate marker XYZ against the warehouse GT in the fixed map frame."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

GT_PATH=Path('labels2coords.json')
# Fixed final GT->map transform from the previous accuracy evaluation AFTER
# the user moved GT zero to the column at map pixel (141, 6298).
# Verified against 391 old evaluated points: max residual < 4e-13 m.
# markers_nizhbel/spatial_nearest_gt_accuracy.csv predates that anchor change
# and MUST NOT supply the translation. No fitting to the new run is done.
GT_AFFINE=np.array([[0.009984473336394264, -0.0005570387731851298],
                    [0.0005570387731851732,  0.009984473336394297]])
GT_OFFSET=np.array([-126.016174, -23.09335737])
MAX_MATCH_M=.60
ROOT=Path('.')
OUT=ROOT/'markers'
MAP_DIR=ROOT/'map'

def read_markers(path):return json.loads(path.read_text())['markers']
def transform_gt(gt):
    labels=list(gt)
    pix=np.array([[gt[k]['x'],gt[k]['y']] for k in labels],float)
    return labels, pix@GT_AFFINE.T+GT_OFFSET
def match(points,gtxy,limit=MAX_MATCH_M):
    distances=np.linalg.norm(points[:,None,:]-gtxy[None,:,:],axis=2)
    # Invalid edges are expensive, so assignment first maximizes valid matches.
    ri,ci=linear_sum_assignment(np.where(distances<=limit,distances,1e6))
    keep=distances[ri,ci]<=limit
    return ri[keep],ci[keep],distances[ri[keep],ci[keep]],distances
def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {k:float(v) for k,v in zip(('median','mean','p90','p95','max'),(np.median(x),np.mean(x),np.percentile(x,90),np.percentile(x,95),np.max(x)))} if len(x) else {}
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--marker-dir',type=Path,required=True)
    parser.add_argument('--map-dir',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    args=parser.parse_args(argv)
    OUT=args.marker_dir;MAP_DIR=args.map_dir;GT_PATH=args.gt
    OUT.mkdir(parents=True,exist_ok=True)
    report_path=OUT/'marker_points_stored_ray_corrected.json'
    doc=json.loads(report_path.read_text());markers=doc['markers'];gt=json.loads(GT_PATH.read_text())
    labels,gtxy=transform_gt(gt)
    points=np.asarray([m['point_xyz_m'][:2] for m in markers],float)
    ri,ci,dd,dist=match(points,gtxy)
    gt_for_marker={int(i):int(j) for i,j in zip(ri,ci)}
    marker_for_gt={int(j):int(i) for i,j in zip(ri,ci)}
    traj_doc=json.loads((MAP_DIR/'trajectory.json').read_text())
    # Trajectory positions from the map export are already in map metric coordinates.
    poses=traj_doc['poses'];track_xy=np.asarray([[p['x'],p['y']] for p in poses],float)
    rows=[]
    for i,m in enumerate(markers):
        j=gt_for_marker.get(i);v={"marker_id":m['marker_id'],"decoder_label":','.join(map(str,m['candidate_codes']))}
        v['x_m']=m['point_xyz_m'][0];v['y_m']=m['point_xyz_m'][1];v['height_m']=m.get('ceiling_z_m')
        v['track_segments']=';'.join(map(str,m['track_segments']));v['height_segments']=';'.join(map(str,m.get('height_segments',[])))
        v['height_excluded_segments']=';'.join(map(str,m.get('height_excluded_segments',[])))
        v['observations']=m['observations'];v['spread_m']=m['spread_m'];v['height_status']=m.get('height_status','estimated')
        v['route_distance_m']=float(np.min(np.linalg.norm(track_xy-points[i],axis=1)))
        if j is not None:
            g=gt[labels[j]];v['matched_gt_label']=labels[j];v['gt_x_m']=gtxy[j,0];v['gt_y_m']=gtxy[j,1]
            v['dx_m']=points[i,0]-gtxy[j,0];v['dy_m']=points[i,1]-gtxy[j,1];v['xy_error_m']=dd[list(ri).index(i)]
            v['gt_height_m']=float(g['h'])/100
            v['height_error_signed_m']=None if m.get('ceiling_z_m') is None else m['ceiling_z_m']-v['gt_height_m']
            v['height_error_abs_m']=None if v['height_error_signed_m'] is None else abs(v['height_error_signed_m'])
        else:
            nearest=int(np.argmin(dist[i]));v['matched_gt_label']='';v['gt_x_m']='';v['gt_y_m']='';v['dx_m']='';v['dy_m']='';v['xy_error_m']=''
            v['gt_height_m']='';v['height_error_signed_m']='';v['height_error_abs_m']='';v['nearest_gt_label']=labels[nearest];v['nearest_gt_distance_m']=float(dist[i,nearest])
        rows.append(v)
    with (OUT/'accuracy_evaluation.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

    # Spatial neighbor evaluation: each matched measured center is compared to
    # the corresponding GT-neighbor distances, using up to 8 GT nearest neighbors.
    pair_rows=[];per_marker=[]
    for center_gt,center_marker in marker_for_gt.items():
        delta=np.linalg.norm(gtxy-gtxy[center_gt],axis=1);neighbor_gt=[j for j in np.argsort(delta) if j!=center_gt][:8]
        errors=[]
        if center_marker not in range(len(markers)):continue
        for j in neighbor_gt:
            neighbor_marker=marker_for_gt.get(int(j))
            if neighbor_marker is None:continue
            gd=float(delta[j]);md=float(np.linalg.norm(points[center_marker]-points[neighbor_marker]));err=md-gd
            errors.append(abs(err));pair_rows.append({'gt_center':labels[center_gt],'measured_center':markers[center_marker]['marker_id'],'gt_neighbor':labels[j],'measured_neighbor':markers[neighbor_marker]['marker_id'],'gt_distance_m':gd,'measured_distance_m':md,'signed_error_m':err,'abs_error_m':abs(err)})
        if errors:
            per_marker.append({'gt_label':labels[center_gt],'measured_marker_id':markers[center_marker]['marker_id'],'neighbor_count':len(errors),'mean_abs_error_m':float(np.mean(errors)),'median_abs_error_m':float(np.median(errors)),'max_abs_error_m':float(np.max(errors))})
    with (OUT/'neighbor_distance_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
    with (OUT/'neighbor_distance_summary_by_marker.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_marker[0]));w.writeheader();w.writerows(per_marker)

    xyerr=[r['xy_error_m'] for r in rows if isinstance(r['xy_error_m'],float)]
    herr=[r['height_error_abs_m'] for r in rows if isinstance(r['height_error_abs_m'],float)]
    neigh=[r['abs_error_m'] for r in pair_rows]
    summary={'markers':len(markers),'matched':len(ri),'unmatched':len(markers)-len(ri),'matched_gt':len(ci),'xy_error_m':stats(xyerr),'height_error_m':stats(herr),'height_n':len(herr),'neighbor_pairs':len(pair_rows),'neighbor_error_m':stats(neigh)}
    (OUT/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Per-marker neighbor mean is the local geometric consistency score.
    nmean={r['measured_marker_id']:r['mean_abs_error_m'] for r in per_marker}
    # Full-map diagnostic: same trajectory and one-to-one GT association for all three metrics.
    fig,axes=plt.subplots(1,3,figsize=(24,10),constrained_layout=True)
    tx,ty=track_xy[:,0],track_xy[:,1]
    for ax in axes:
        ax.plot(tx,ty,color='0.65',linewidth=.45,alpha=.75,zorder=1)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    matched_ids=set(gt_for_marker)
    xyvals=[float(rows[i]['xy_error_m']) for i in matched_ids]
    sc=axes[0].scatter(points[list(matched_ids),0],points[list(matched_ids),1],c=xyvals,s=10,cmap='magma',vmin=0,vmax=max(.30,max(xyvals,default=.30)),zorder=3)
    fig.colorbar(sc,ax=axes[0],label='XY error (m)');axes[0].set_title(f"Position error, n={len(xyvals)}")
    hidx=[i for i in matched_ids if isinstance(rows[i]['height_error_abs_m'],float)]
    hv=[float(rows[i]['height_error_abs_m']) for i in hidx]
    sc=axes[1].scatter(points[hidx,0],points[hidx,1],c=hv,s=10,cmap='viridis',vmin=0,vmax=max(.30,max(hv,default=.30)),zorder=3)
    fig.colorbar(sc,ax=axes[1],label='|height error| (m)');axes[1].set_title(f"Height error, n={len(hv)}")
    ni=[i for i,m in enumerate(markers) if m['marker_id'] in nmean]
    nv=[nmean[markers[i]['marker_id']] for i in ni]
    sc=axes[2].scatter(points[ni,0],points[ni,1],c=nv,s=10,cmap='plasma',vmin=0,vmax=max(.30,max(nv,default=.30)),zorder=3)
    fig.colorbar(sc,ax=axes[2],label='Mean |neighbor distance error| (m)');axes[2].set_title(f"Neighbor geometry (up to 8), n={len(nv)}")
    fig.suptitle(f'Marker accuracy on {MAP_DIR.parent.name} — fixed warehouse GT frame')
    fig.savefig(OUT/'accuracy_evaluation.png',dpi=180);plt.close(fig)
    def metric_map(filename,title,indices,values,cbar):
        vals=np.asarray(values,float)
        fig,ax=plt.subplots(figsize=(13,11),constrained_layout=True)
        ax.plot(tx,ty,color='0.65',linewidth=.5,alpha=.8,zorder=1)
        sc=ax.scatter(points[indices,0],points[indices,1],c=vals,s=15,cmap='turbo',vmin=0,vmax=max(.30,float(vals.max()) if len(vals) else .30),zorder=3)
        fig.colorbar(sc,ax=ax,label=cbar)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_title(title)
        st=stats(vals)
        box='n={n}\nmean {mean:.1f} cm | median {median:.1f} cm\np90 {p90:.1f} cm | p95 {p95:.1f} cm\nmax {max:.1f} cm'.format(n=len(vals),**{k:v*100 for k,v in st.items()})
        ax.text(.015,.985,box,transform=ax.transAxes,va='top',ha='left',fontsize=11,bbox={'facecolor':'white','alpha':.88,'edgecolor':'0.4'})
        fig.savefig(OUT/filename,dpi=180);plt.close(fig)
    xyidx=sorted(matched_ids);xyv=[float(rows[i]['xy_error_m']) for i in xyidx]
    metric_map('map_position_error.png','Position error vs GT (one-to-one spatial matching)',xyidx,xyv,'XY error (m)')
    metric_map('map_height_error.png','Absolute marker-height error vs GT',hidx,hv,'|height error| (m)')
    metric_map('map_neighbor_error.png','Mean absolute distance error to up to 8 nearest GT neighbors',ni,nv,'Mean |distance error| (m)')
    unmatched=[r for r in rows if r['matched_gt_label']=='']
    with (OUT/'unmatched_markers.csv').open('w',newline='') as f:
        if unmatched:
            w=csv.DictWriter(f,fieldnames=list(unmatched[0]));w.writeheader();w.writerows(unmatched)
    def line(name,d): return f"- {name}: median **{d['median']*100:.1f} cm**, mean **{d['mean']*100:.1f} cm**, p90 **{d['p90']*100:.1f} cm**, p95 **{d['p95']*100:.1f} cm**, max **{d['max']*100:.1f} cm**"
    report=['# Оценка перенесённых меток — 20261029_004258_nav_video','',
        f"Сопоставлено по положению один-к-одному в радиусе {MAX_MATCH_M:.2f} м: {len(ri)} из {len(markers)}; не сопоставлено: {len(markers)-len(ri)}.",
        'GT преобразован в систему карты по зафиксированному преобразованию предыдущей проверенной оценки; новые точки не использовались для подгонки привязки.', '',
        '## Ошибка положения XY','',line('XY',summary['xy_error_m']),'',
        '## Ошибка высоты','',line('Абсолютная ошибка высоты',summary['height_error_m']),
        f"Высоты рассчитаны для {summary['height_n']} пространственно сопоставленных точек; сравнение с полем h из labels2coords.json (см → м).",'',
        '## Геометрия соседей','',line('Абсолютная ошибка расстояния',summary['neighbor_error_m']),
        f"Всего {len(pair_rows)} направленных сравнений: для каждой GT-метки брались до 8 ближайших GT-соседей; расстояния сравнивались только если обе точки имеют пространственно сопоставленные измерения. Метрика карты — средняя абсолютная ошибка соседних расстояний для каждой центральной метки.", '',
        '## Карты','',
        '- `map_position_error.png` — XY;','- `map_height_error.png` — абсолютная ошибка высоты;','- `map_neighbor_error.png` — средняя ошибка расстояний до соседей;','- `accuracy_evaluation.png` — все три показателя рядом с траекторией.','',
        'Таблицы: `accuracy_evaluation.csv`, `neighbor_distance_errors.csv`, `neighbor_distance_summary_by_marker.csv`, `unmatched_markers.csv`.']
    (OUT/'evaluation_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
