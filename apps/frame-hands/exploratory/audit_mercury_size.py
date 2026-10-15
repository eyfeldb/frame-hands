"""Audit native size-refinement state and paired frozen-observation outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def stats(x):return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','observations','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();data=json.loads(a.summary.read_text());source=json.loads(a.observations.read_text());contexts={r['case']['name']:r for r in source['reports']};groups={};verified=0
    for group in sorted({r['case']['name'].rsplit('-',1)[0] for r in data['reports']}):
        frames=[r for r in data['reports'] if r['case']['name'].rsplit('-',1)[0]==group];expected_x=np.float32(0);expected_size=np.float32(.09)
        for row in frames:
            pre=row['scale_pre'];post=row['scale_post'];np.testing.assert_allclose(pre['schedule_x'],expected_x,atol=2e-6);np.testing.assert_allclose(pre['target_size'],expected_size,atol=1e-8)
            expected_weight=np.float32(np.float32(expected_x/np.float32(100))**2);np.testing.assert_allclose(pre['size_error'],expected_weight,atol=2e-8)
            fit=row.get('arms',{}).get('native_size',{})
            if fit.get('accepted'):
                original=contexts[row['case']['name']];views={v['camera']:v['feedback'] for v in original['views']};total=np.float32(0)
                for name in ('slam_left','slam_right'):
                    for confidence in views[name]['confidence_xy']:total=np.float32(total+np.float32(confidence))
                mean=np.float32(total/np.float32(42));error=np.float32(fit['fit']['reprojection_error']);quality=np.float32(mean*np.float32(1/np.float32(error*np.float32(10)+np.float32(1))))
                expected_x=np.float32(expected_x+quality);expected_size=np.float32(fit['fit']['hand_size_m']);verified+=1
            np.testing.assert_allclose(post['schedule_x'],expected_x,atol=2e-6);np.testing.assert_allclose(post['target_size'],expected_size,atol=1e-8)
        accepted=[r for r in frames if r.get('arms',{}).get('native_size',{}).get('accepted')];arms={}
        for branch in ('fixed','native_size'):
            values=[r['arms'][branch] for r in accepted];sizes=[v['fit']['hand_size_m']*1000 for v in values];disagreement=[x['median'] for v in values for x in v['ncnn_disagreement_px'].values()];steps=[]
            for ra,rb in zip(accepted[:-1],accepted[1:]):
                if 0<rb['stamp']-ra['stamp']<=100000000:steps.append(float(np.max(np.linalg.norm(np.asarray(rb['arms'][branch]['world_joints_mm'])-ra['arms'][branch]['world_joints_mm'],axis=1))))
            arms[branch]=dict(accepted=len(values),size_min_mm=float(min(sizes)),size_max_mm=float(max(sizes)),largest_size_step_mm=float(max(abs(np.diff(sizes)))),ncnn_disagreement_px=stats(disagreement),max_joint_step_mm=stats(steps),wrist_range_mm=stats([np.linalg.norm(v['fit']['joints_left_camera_m'][1])*1000 for v in values]))
        shifts=[np.linalg.norm(np.asarray(r['arms']['fixed']['world_joints_mm'])[0]-np.asarray(r['arms']['native_size']['world_joints_mm'])[0]) for r in accepted]
        groups[group]=dict(arms=arms,final_schedule_x=frames[-1]['scale_post']['schedule_x'],calibration_complete=not frames[-1]['scale_post']['optimize_size'],paired_wrist_shift_mm=stats(shifts))
    out=dict(verified_native_schedule_updates=verified,exact_fixed_baseline_fits=data['exact_fixed_baseline_fits'],groups=groups,input_sha256=hashlib.sha256(a.summary.read_bytes()).hexdigest(),observations_sha256=hashlib.sha256(a.observations.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Independent one-second single-hand bootstrap windows never complete native calibration. Size changes can compensate inaccurate2Dpoints. No calibrated human dimensions or3Dground truth. Image disagreement and motion are proxies. Learned observations/crops are frozen from90mm feedback; changing scale does not alter them here.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print('Verified schedule updates:',verified)
    for group,g in groups.items():print(group,'wrist shift',g['paired_wrist_shift_mm'],'complete',g['calibration_complete'])

if __name__=='__main__':main()
