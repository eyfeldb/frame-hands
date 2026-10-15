"""Independent known-pose and lifecycle audit of bounded correspondence continuity."""
import argparse,json,sys,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from audit_native_lengths import sha,stats
from audit_finger_correspondence import disagreement,motion
from audit_host_tracker import read_trace
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,read_audit


def synthetic(folder,reference,profiles,evaluator,renewal=False):
    data=np.loadtxt(folder/'output.txt');old=np.loadtxt(reference/('output.txt' if renewal else 'output-v2.txt'));assert data.shape==(1176,101 if renewal else 100)
    if renewal:
        disabled=np.loadtxt(folder/'control.txt');assert np.array_equal(np.delete(disabled,9,1),np.delete(old,9,1))
    lookup={tuple(r[:4].astype(int)):r for r in old};controls=0
    for row in data:
        if renewal or row[2]<5:
            assert np.array_equal(row[43:70],lookup[tuple(row[:4].astype(int))][70:97]);controls+=1
    points=np.empty((len(data),3,21,3))
    for side in (0,1):
        selected=data[:,1]==side;poses=data[selected,16:97].reshape(-1,27)
        input_file=folder/f'audit-side{side}-input.txt';output_file=folder/f'audit-side{side}-points.txt'
        with input_file.open('x') as stream:np.savetxt(stream,np.c_[np.full(len(poses),side),poses],fmt='%.17g')
        subprocess.run([str(evaluator.resolve()),str((profiles/f'side{side}.fhrig').resolve()),str(input_file.resolve()),str(output_file.resolve())],check=True,capture_output=True,text=True)
        geometry=np.loadtxt(output_file);assert geometry.shape==(len(poses),64);points[selected]=geometry[:,1:].reshape(-1,3,21,3)
    distances=np.linalg.norm(points[:,1:]-points[:,:1],axis=-1);errors=np.sqrt(np.mean(distances**2,axis=2));actual=data[:,[10,13]]
    difference=float(np.max(abs(actual-errors)));assert difference<1e-10
    groups=[]
    for family in (0,1):
        for mode in range(6):
            selected=(data[:,0]==family)&(data[:,2]==mode);x=data[selected];e=errors[selected]
            if mode<5:assert np.array_equal(x[:,43:70],x[:,70:97])
            groups.append(dict(family=('independent','causal_sequence')[family],mode=('clean','noise2px','swap_left','swap_right','swap_both','swap_left_with_weak_interval')[mode],samples=len(x),continued=int(sum(x[:,97]>0)),renewed=int(sum(x[:,100]>0)) if renewal else 0,truth_rmse_mm=[stats(e[:,k]) for k in (0,1)],improvements_over1mm=int(sum(e[:,0]-e[:,1]>1)),regressions_over1mm=int(sum(e[:,1]-e[:,0]>1))))
    result=dict(cases=len(data),old_controls_exact=controls,comparison="bounded_vs_renewal" if renewal else "strict_vs_bounded",max_reconstructed_error_difference=difference,groups=groups,output_sha256=sha(folder/'output.txt'),limitations=['Shared-model geometry. Only first sequence frame receives truth initialization; later poses are each arm own previous fit.','Weak interval is a declared24px vertical offset on one swapped finger pair for four frames, not calibrated physical noise.','No image inference, visibility, native performance or wearer accuracy claim.'])
    (folder/'synthetic-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


def verify_lifecycle(frames,records,head_rows,renewal=False):
    state={};generation=None;continued=0;renewals=0;max_age=0
    for frame in frames:
        raw=frame['raw'];head=head_rows[raw];head_valid=head['valid'];new_generation=head['generation']
        if not head_valid or generation!=new_generation:state={}
        generation=new_generation if head_valid else None
        counts=Counter(row['side'] for row in frame['rows'].values());now={}
        for source,row in frame['rows'].items():
            if row['status']!='fitted':continue
            observed=records[(raw,source)];side=row['side'];key=(source,side);old=state.get(key)
            original=observed['original'];alternatives=observed['alternatives'];choice=observed['choice'];carried=observed['continued'];age=observed['age'];strong=observed['strong'];continuous=head_valid and counts[side]==1
            assert choice in (-1,0,1)
            if carried:
                assert continuous and old is not None and choice==old['choice']
                assert 0<raw-old['raw']<=100000000 and 0<raw-old['strong']<=100000000
                assert np.isfinite(original) and original>3 and 3<alternatives[choice]<original
                should_renew=renewal and original-alternatives[choice]>=3
                assert observed.get('renewed',False)==should_renew
                assert age==raw-old['strong'] and strong==(raw if should_renew else old['strong'])
                renewals+=should_renew
                continued+=1;max_age=max(max_age,age)
            else:
                assert age==0 and not observed.get('renewed',False)
                if choice>=0:
                    assert np.isfinite(original) and original>3 and np.isfinite(alternatives[choice]) and alternatives[choice]<=3
                    assert strong==(raw if continuous else 0)
                else:assert strong==0
            if choice>=0:
                assert abs(observed['costs'][choice]-row['values'][0])<1e-8
                if continuous:now[key]=dict(raw=raw,strong=strong,choice=choice)
        state=now
    return dict(continued=continued,renewed=int(renewals),max_evidence_age_ns=max_age)


def replay(folder,session,renewal=False):
    cameras=load_calibration(session/'frame/snapshot/device/persist');views=read_audit(folder/'1-candidate/views.txt')
    records={};heads={};raw=None
    for line in (folder/'1-candidate/rig.txt').read_text().splitlines():
        fields=line.split()
        if fields[0]=='frame':raw=int(fields[1])
        elif fields[0]=='head_recorded':heads[int(fields[1])]=dict(generation=int(fields[5]),valid=bool(int(fields[6])))
        elif fields[0]=='correspondence':
            source,choice=int(fields[1]),int(fields[2]);numbers=np.array(fields[3:],float)
            records[(raw,source)]=dict(choice=choice,original=numbers[0],alternatives=numbers[1:3].tolist(),costs=numbers[3:].tolist())
        elif fields[0]=='continuity':records[(raw,int(fields[1]))].update(continued=bool(int(fields[2])),age=int(fields[3]),strong=int(fields[4]))
        elif fields[0]=='renewal':records[(raw,int(fields[1]))]['renewed']=bool(int(fields[2]))
    differences=[]
    for (raw,source),record in records.items():
        seen=[next(v for v in views[raw]['views'] if v['id']==source and v['camera']==name) for name in ('slam_left','slam_right')];points=[v['points'].copy() for v in seen]
        original=disagreement(cameras,points);expected=[original,np.inf,np.inf]
        if np.isfinite(original) and original>3:
            for view in (0,1):
                alternative=[p.copy() for p in points];alternative[view][5:9],alternative[view][9:13]=points[view][9:13],points[view][5:9];expected[1+view]=disagreement(cameras,alternative)
        expected=np.array(expected);actual=np.array([record['original'],*record['alternatives']]);finite=np.isfinite(expected);assert np.array_equal(finite,np.isfinite(actual));difference=float(np.max(abs(expected[finite]-actual[finite]))) if np.any(finite) else 0;assert difference<1e-4;differences.append(difference)
    frames=read_trace(folder/'1-candidate/rig.txt');result=verify_lifecycle(frames,records,heads,renewal)
    # Forged renewed deadline must be detected independently of geometry.
    if result['continued']:
        import copy
        altered=copy.deepcopy(records);bad=next(v for v in altered.values() if v['continued']);bad['age']=0
        try:verify_lifecycle(frames,altered,heads,renewal)
        except AssertionError:pass
        else:raise AssertionError('Renewed weak deadline accepted')
    if result['renewed']:
        altered=copy.deepcopy(records);bad=next(v for v in altered.values() if v.get('renewed'));bad['alternatives'][bad['choice']]=bad['original']-2
        try:verify_lifecycle(frames,altered,heads,renewal)
        except AssertionError:pass
        else:raise AssertionError('Unsupported renewal accepted')
    expected={(frame['raw'],source) for frame in frames for source,row in frame['rows'].items() if row['status']=='fitted'}
    assert expected and set(records)==expected
    result.update(rows=len(records),choices=[sum(r['choice']==v for r in records.values()) for v in (-1,0,1)],max_python_geometry_difference=max(differences),negative_controls=int(result['continued']>0)+int(result['renewed']>0),trace_sha256=sha(folder/'1-candidate/rig.txt'),source_sha256=sha(Path(__file__)))
    (folder/'continuity-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('synthetic','reference','profiles','evaluator','run','session','upstream','heads'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--renewal',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(args.upstream))
    a=synthetic(args.synthetic,args.reference,args.profiles,args.evaluator,args.renewal);b=replay(args.run,args.session,args.renewal);motion(args.run,args.heads)
    print(json.dumps(dict(synthetic_cases=a['cases'],old_controls=a['old_controls_exact'],replay=b),indent=2))


if __name__=='__main__':main()
