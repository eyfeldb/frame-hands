"""Independent output audit for the frozen native-feedback experiment."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def stats(x):return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('feedback','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();data=json.loads(a.feedback.read_text());base=json.loads(a.baseline.read_text());history=[];last_group=None;verified=0;exact=0
    for row,old in zip(data['reports'],base['reports'],strict=True):
        if row['stamp']!=old['stamp']:raise AssertionError('Capture mismatch')
        group=row['case']['name'].rsplit('-',1)[0]
        if group!=last_group:history=[];last_group=group
        if 'skipped' in row:history=[];continue
        source=[r[0] for r in history]
        if source!=row['prior_source_stamps']:raise AssertionError('History source mismatch')
        expected=None
        if len(history)==2:
            (ta,ga,pa),(tb,gb,pb)=history;stamp=row['stamp']
            if ga==gb==row['generation'] and ta<tb<stamp and stamp-tb<=100000000:expected=pb+(pb-pa)*(.4*(stamp-tb)/(tb-ta))
        actual=row['prior_world_mm']
        if expected is None:
            if actual is not None:raise AssertionError('Unexpected prior')
        else:
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0);verified+=1
        new=row['arms']['off'];prior=old['arms']['off-compensated']
        if new['fit']!=prior['fit'] or new['world_joints_mm']!=prior['world_joints_mm']:raise AssertionError('Baseline fit changed')
        exact+=1;feedback=row['arms']['feedback']
        if feedback.get('accepted'):
            history.append((row['stamp'],row['generation'],np.asarray(feedback['world_joints_mm'])));history=history[-2:]
        else:history=[]
    views=[v for r in data['reports'] for v in r.get('views',[])];counts={k:sum(v[k]['active'] for v in views) for k in ('off','feedback','background_off','background_feedback','blank')};groups={}
    for group in sorted({r['case']['name'].rsplit('-',1)[0] for r in data['reports']}):
        frames=[r for r in data['reports'] if r['case']['name'].rsplit('-',1)[0]==group];branches={}
        for branch in ('off','feedback'):
            accepted=[r for r in frames if r.get('arms',{}).get(branch,{}).get('accepted')];steps=[];disagreement=[]
            for prev,row in zip(accepted[:-1],accepted[1:]):
                if 0<row['stamp']-prev['stamp']<=100000000 and row['generation']==prev['generation']:steps.append(float(np.max(np.linalg.norm(np.asarray(row['arms'][branch]['world_joints_mm'])-prev['arms'][branch]['world_joints_mm'],axis=1))))
            for row in accepted:disagreement.extend(v['median'] for v in row['arms'][branch]['ncnn_disagreement_px'].values())
            branches[branch]=dict(accepted=len(accepted),max_joint_step_mm=stats(steps),median_joint_ncnn_disagreement_px=stats(disagreement))
        groups[group]=branches
    false_positives=[dict(case=r['case']['name'],seconds=r['seconds'],camera=v['camera'],prior_used=v['prior_used'],unconditioned_presence=v['background_off']['presence'],conditioned_presence=v['background_feedback']['presence']) for r in data['reports'] for v in r.get('views',[]) if v['background_feedback']['active']]
    out=dict(exact_baseline_fits=exact,verified_causal_predictions=verified,camera_views=len(views),prior_used=sum(v['prior_used'] for v in views),counts=counts,paired_background_pass=sum(all(v['background_feedback']['active'] for v in r['views']) for r in data['reports'] if 'views' in r),groups=groups,false_positive_contexts=false_positives,input_sha256=hashlib.sha256(a.feedback.read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Motion/disagreement are proxies, not true accuracy. Two fixed background textures reused across contexts are not independent negative images. Controls borrow a real prior for one frame and never update feedback history. Stereo-only admission does not establish safe monocular fallback.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print('Exact fits',exact,'causal predictions',verified,'counts',counts)

if __name__=='__main__':main()
