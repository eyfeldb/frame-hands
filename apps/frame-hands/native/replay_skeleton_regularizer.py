"""Causal saved-observation test of soft anatomy against fresh stereo triangulation.
Isolates reconstruction before filtering. Neither branch is the accepted driver output.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from compare_replay import source_hands
from audit_native_lengths import PARENTS,FINGERS,TIPS,sha,stats,errors
from stereo_uncertainty import sensitivity
from skeleton_regularizer import fit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('session','upstream','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream))
    if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
    a.output.mkdir(parents=True,exist_ok=False);source=source_hands(a.session)
    audit=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist')
    histories={};previous={};rows=[];metrics={k:[] for k in ('control_tip_error_px','candidate_tip_error_px','max_correction_mm','control_length_step_mm','candidate_length_step_mm','fit_ms','constraint_count')}
    counts=dict(stereo=0,fit=0,rejected=0,unchanged=0,geometry_skipped=0);prior_errors=[]
    with (a.output/'poses.jsonl').open('x') as out:
        for stamp,ob in sorted(audit.items()):
            for ident,h in source.get(ob['mono_ns'],{}).items():
                views=[v for v in ob['views'] if v['id']==ident and v['camera'] in cams]
                if not h['stereo'] or len(views)!=2 or len({v['camera'] for v in views})!=2:continue
                try:x,cov,info=sensitivity(cams,views)
                except ValueError:counts['geometry_skipped']+=1;continue
                counts['stereo']+=1;key=(ident,h['side']);old=histories.get(key)
                if old is None or not 0<stamp-old['stamp']<=80_000_000:old=dict(stamp=stamp,samples=[[] for _ in range(21)],lengths=np.full(21,np.nan));histories[key]=old
                old['stamp']=stamp
                # Use prior samples only. Current data cannot create its own constraint.
                begin=time.perf_counter();y,debug=fit(x,cov,old['lengths']);metrics['fit_ms'].append((time.perf_counter()-begin)*1000)
                e0=errors(x,views,cams);e1=errors(y,views,cams)
                correction=float(np.max(np.linalg.norm(y-x,axis=1)));length=np.linalg.norm(y-y[PARENTS],axis=1)
                admitted=bool(e0 is not None and e1 is not None and np.all(e1.mean(axis=1)<=e0.mean(axis=1)+2) and correction<=50 and np.all(length[1:]>=3) and np.all(length[1:]<=160))
                if debug['active_bones']:
                    counts['fit']+=1
                    if not admitted:counts['rejected']+=1;y=x.copy();e1=e0;correction=0
                else:counts['unchanged']+=1
                metrics['constraint_count'].append(debug['active_bones']);metrics['max_correction_mm'].append(correction)
                if e0 is not None and e1 is not None:
                    metrics['control_tip_error_px'].append(float(e0[:,TIPS].mean()));metrics['candidate_tip_error_px'].append(float(e1[:,TIPS].mean()))
                rawlength=np.linalg.norm(x-x[PARENTS],axis=1);delta=x[FINGERS]-x[PARENTS[FINGERS]];u=delta/np.maximum(rawlength[FINGERS,None],1e-8)
                sigma=np.sqrt(np.maximum(0,np.einsum('ni,nij,nj->n',u,cov[FINGERS]+cov[PARENTS[FINGERS]],u)))
                good=(min(v['presence'] for v in views)>=.8 and np.median(info['reprojection_px'])<=8 and np.percentile(info['reprojection_px'],95)<=20)
                for i,s in zip(FINGERS,sigma):
                    if good and s<=.1*rawlength[i] and 3<=rawlength[i]<=160 and not np.isfinite(old['lengths'][i]):
                        samples=old['samples'][i];samples.append(rawlength[i]);samples[:]=samples[-15:]
                        if len(samples)==15:
                            m=np.median(samples)
                            if np.median(np.abs(np.array(samples)-m))<=.1*m:old['lengths'][i]=m
                lengths=np.stack([rawlength,np.linalg.norm(y-y[PARENTS],axis=1)])
                last=previous.get(key)
                if last and 0<stamp-last[0]<=80_000_000:
                    for j,label in enumerate(('control','candidate')):metrics[label+'_length_step_mm'].extend(np.abs(lengths[j,FINGERS]-last[1][j,FINGERS]).tolist())
                previous[key]=(stamp,lengths)
                row=dict(raw_ns=stamp,source_id=ident,side=h['side'],active_bones=debug['active_bones'],admitted=admitted,control_head_mm=x.tolist(),candidate_head_mm=y.tolist(),tip_error_change_px=float(e1[:,TIPS].mean()-e0[:,TIPS].mean()) if e0 is not None and e1 is not None else None,max_change_mm=correction)
                out.write(json.dumps(row)+'\n');rows.append(row)
    report=dict(scope=__doc__,session=a.session.name,counts=counts,metrics={k:stats(v) for k,v in metrics.items()},parameters=dict(pixel_sigma_assumed=3,anatomy_sigma='max(2mm,10%length)',iterations=5,calibration_samples=15,calibration_sigma_at_1px_max_fraction=.10,guard_max_mean_reprojection_increase_px_per_view=2,guard_max_correction_mm=50),provenance=dict(inputs={n:sha(a.session/n) for n in ('relay-trace.bin','frame/live-views.txt')},sources={n:sha(BASE/n) for n in ('stereo_uncertainty.py','skeleton_regularizer.py','replay_skeleton_regularizer.py')}),limitations=['Pixel covariance is a noise assumption, not calibrated detector certainty.','Reprojection uses fitting observations, not independent accuracy labels.','Neither control nor candidate includes accepted temporal filter; no claims about final display jitter.','No identity/recovery changes or native device timing.'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    changed=[r for r in rows if r['max_change_mm']>.01 and r['tip_error_change_px'] is not None]
    chosen=sorted(changed,key=lambda r:r['max_change_mm'],reverse=True)[:2]
    (a.output/'review-selection.json').write_text(json.dumps(chosen,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
