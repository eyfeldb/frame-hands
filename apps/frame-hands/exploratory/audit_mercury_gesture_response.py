"""Independently check synthetic gesture inputs, metrics, and recorded-fit parity."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from mercury_forward import NativeHandGenerator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('result','parity','dll','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();summary=json.loads((args.result/'summary.json').read_text());parity=json.loads((args.parity/'summary.json').read_text())
    digest=hashlib.sha256(args.dll.read_bytes()).hexdigest()
    assert digest==summary['dll_sha256']==parity['dll_sha256']
    assert parity['counts']['exact_total']==4464
    for name,digest in summary['source_sha256'].items():assert hashlib.sha256((args.result/'snapshot'/name).read_bytes()).hexdigest()==digest
    generator=NativeHandGenerator(args.dll);reports=[];count=0;projection_max_error=0.
    mapping=[1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]
    begin=np.array([-.1,-.05,-.05]);end=np.array([-1.2,-1.2,-.8]);wrist=[.70710678,0,0,.70710678,.04,-.03,-.5]
    for case in summary['reports']:
        side=case['side'];duration=case['duration_frames'];noisy=case['condition']=='gaussian'
        rows=[json.loads(line) for line in (args.result/(case['case']+'.jsonl')).read_text().splitlines()];assert len(rows)==180
        rng=np.random.default_rng(20261005+100*side+duration);arms={}
        for index,row in enumerate(rows):
            assert row['index']==index and row['seconds']==index/30
            fraction=min(1,max(0,(index-59)/duration))-min(1,max(0,(index-109)/duration))
            assert abs(fraction-row['truth_fraction'])<1e-15
            angles=np.tile(begin+(end-begin)*fraction,(4,1));np.testing.assert_array_equal(angles,row['truth_angles'])
            truth=generator.generate(wrist,angles,side)[mapping,:3];np.testing.assert_array_equal(truth,row['truth_joints_m'])
            for name,offset in [('slam_left',0),('slam_right',-.06)]:
                # Independent analytic stereographic projection; equivalent to unit-ray form.
                points=truth.astype(float).copy();points[:,0]+=offset
                uv=points[:,:2]/(np.linalg.norm(points,axis=1)-points[:,2])[:,None]/.4
                uv[:,1]*=-1
                if side:uv[:,0]*=-1
                pixels=(uv+1)*64;curls=angles.sum(axis=1).copy()
                if noisy:pixels+=rng.normal(0,.5,(21,2));curls+=rng.normal(0,.4,4)
                view=row['views'][name]
                # Producer uses float32 arithmetic in the right-camera projection.
                projection_max_error=max(projection_max_error,float(abs(pixels-np.asarray(view['points_crop'])).max()))
                np.testing.assert_allclose(pixels,view['points_crop'],rtol=0,atol=1.6e-5)
                np.testing.assert_array_equal(curls,view['curls'][1:5]);np.testing.assert_array_equal(view['curls'][6:10],[.15 if noisy else 0]*4)
                np.testing.assert_array_equal(view['depth_heatmap'],np.zeros((21,22)))
                np.testing.assert_array_equal(view['confidence_xy'],np.ones(21));np.testing.assert_array_equal(view['rotation'],np.eye(3))
                assert view['active'] and view['stereographic_radius']==.4
            for record in row['arms'].values():
                joints=np.asarray(record['joints_m']);fingers=np.asarray(record['angles'])
                error=np.linalg.norm(joints-truth,axis=1)*1000
                assert abs(error.max()-record['max_joint_error_mm'])<1e-10
                assert abs(error.mean()-record['mean_joint_error_mm'])<1e-10
                assert abs(abs(fingers-angles).max()-record['max_angle_error_rad'])<1e-12
                # Fitted arrays were float32 in the producer; allow only rounding here.
                progress=(begin.sum()-fingers.sum(axis=1).mean())/(begin.sum()-end.sum())
                assert abs(progress-record['estimated_fraction'])<2e-7
                count+=1
        for name,claimed in case['arms'].items():
            values=[row['arms'][name] for row in rows];accepted=sum(v['accepted'] for v in values)
            assert accepted==claimed['accepted']
            groups={'initial_settled_max_joint_error_mm':values[40:60],
                    'settled_max_joint_error_mm':values[40:60]+values[150:180],
                    'moving_max_joint_error_mm':[v for i,v in enumerate(values) if 60<=i<60+duration or 110<=i<110+duration]}
            for metric,group in groups.items():
                data=[v['max_joint_error_mm'] for v in group];actual=[len(data),float(np.median(data)),float(np.percentile(data,95)),max(data)]
                np.testing.assert_allclose(actual,[claimed[metric][key] for key in ('count','median','p95','maximum')],rtol=0,atol=1e-10)
            responses={}
            for event,start,stop,sign in [('close',60,110,1),('open',110,180,-1)]:
                truth=next(i for i in range(start,stop) if sign*(rows[i]['truth_fraction']-.5)>=0)
                fitted=next((i for i in range(start,stop) if sign*(values[i]['estimated_fraction']-.5)>=0),None)
                result=dict(truth_frame=truth,estimated_frame=fitted,delay_ms=None if fitted is None else (fitted-truth)*1000/30)
                assert truth==claimed['response'][event]['truth_frame'] and fitted==claimed['response'][event]['estimated_frame']
                np.testing.assert_allclose(result['delay_ms'],claimed['response'][event]['delay_ms'],atol=1e-12)
                responses[event]=result
            arms[name]=dict(accepted=accepted,initial_p95_mm=claimed['initial_settled_max_joint_error_mm']['p95'],moving_p95_mm=claimed['moving_max_joint_error_mm']['p95'],response=responses)
        reports.append(dict(case=case['case'],arms=arms))
    assert count==2880 and len(reports)==8
    report=dict(verified_fits=count,projection_roundoff_max_pixels=projection_max_error,recorded_fits_exact=parity['counts']['exact_total'],independent_checks=['forward truth','analytic stereo projection','declared noise schedule','pose error','threshold response','source and DLL hashes'],reports=reports,limitations='Synthetic truth uses the same native geometry family as the inverse solver. Thirty-Hz threshold delay is not end-to-end latency. Initial inverse-fit error is nonzero even for clean observations; noisy right slow trial has large bias. No image-model, acquisition, identity or physical accuracy claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
