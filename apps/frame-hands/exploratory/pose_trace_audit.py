"""Classify exact saved capture-pose availability without fabricating missing poses.

Missing relay/sampler evidence is not an observed headset tracking failure.
This module is offline-only and never retimes, interpolates or publishes a pose.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
import numpy as np


def records(path, size, magic):
    with Path(path).open('rb') as stream:
        while data := stream.read(size):
            if len(data) != size or struct.unpack_from('<8sII',data) != (magic,1,size):
                raise ValueError('Malformed or truncated trace record')
            yield data


def valid_matrix(values):
    xf=np.asarray(values,float).reshape(3,4)
    rotation=xf[:,:3]
    return bool(np.isfinite(xf).all() and
        np.allclose(rotation.T@rotation,np.eye(3),atol=1e-3) and
        abs(np.linalg.det(rotation)-1)<1e-3)


def classify(exports, samples, usable_sequences):
    """Classify evidence, not the unobserved physical tracking state."""
    if not exports:return 'not_exported'
    if not samples:return 'not_sampled'
    good=[s for s in samples if s['capture_flags']&3==3 and s['matrix_valid']]
    if any(s['sequence'] in usable_sequences for s in good):return 'exact_join_valid'
    if good:return 'unusable_transport_pose'
    if not any(s['flags']&2 for s in samples):return 'mailbox_stale'
    if not any(s['head_flags']&3==3 for s in samples):return 'current_head_invalid'
    if not any(s['capture_flags']&3==3 for s in samples):return 'capture_head_invalid'
    return 'matrix_rejected'


def load_context(session):
    # Reuse the actual saved-protocol conversion criteria, without calling main.
    path=Path(__file__).resolve().parent.parent/'pc-input/relay.py'
    spec=importlib.util.spec_from_file_location('hand_trace_relay_conversion',path)
    relay=importlib.util.module_from_spec(spec);spec.loader.exec_module(relay)
    exports=collections.defaultdict(list);seq_capture={};usable=set();relay_rows=[]
    for data in records(session/'relay-trace.bin',3280,b'FHRELAY1'):
        if data[2712:2720]!=b'FHPCVR1\0':raise ValueError('Malformed relay PC packet')
        start,end,reused,published=struct.unpack_from('<QQII',data,16)
        sequence=struct.unpack_from('<Q',data,2728)[0]
        _,stat=relay.convert(data[40:2712],start,end,sequence)
        capture=stat['source_capture']
        if sequence in seq_capture and seq_capture[sequence]!=capture:
            raise ValueError('One relay sequence names different source captures')
        seq_capture[sequence]=capture
        if stat['valid']:usable.add(sequence)
        row=dict(sequence=sequence,start_ms=start,end_ms=end,reused=bool(reused),published=bool(published),
                 transport_valid=stat['valid'],capture_mono_ns=capture)
        exports[capture].append(row);relay_rows.append(row)
    samples=collections.defaultdict(list);sampler_rows=[]
    for data in records(session/'sampler-trace.bin',3136,b'FHTRACE1'):
        time,flags,head,capture=struct.unpack_from('<QIII',data,16)
        sequence=struct.unpack_from('<Q',data,720)[0]
        matrix=struct.unpack_from('<12f',data,88)
        row=dict(time_ms=time,flags=flags,head_flags=head,capture_flags=capture,
                 sequence=sequence,matrix_valid=valid_matrix(matrix),matrix=matrix)
        samples[sequence].append(row);sampler_rows.append(row)
    return exports,samples,usable,relay_rows,sampler_rows


def audit_session(session):
    from frame_data import read_audit,capture_poses
    observations=read_audit(session/'frame/live-views.txt');existing=capture_poses(session)
    exports,samples,usable,relays,sampler=load_context(session)
    first=min(observations);rows=[];counts=collections.Counter();differences=[]
    for raw,obs in observations.items():
        capture=obs['mono_ns'];exported=exports.get(capture,[])
        selected=[s for seq in {e['sequence'] for e in exported} for s in samples.get(seq,[])]
        reason=classify(exported,selected,usable);counts[reason]+=1
        if (capture in existing)!=(reason=='exact_join_valid'):differences.append(raw)
        rows.append(dict(seconds=(raw-first)/1e9,capture_raw_ns=raw,capture_mono_ns=capture,
            reason=reason,export_records=len(exported),sampler_records=len(selected),
            observed_flags=sorted({(s['flags'],s['head_flags'],s['capture_flags']) for s in selected})))
    intervals=[];current=None
    for row in rows:
        if row['reason']=='exact_join_valid':current=None;continue
        if current is None or current['reason']!=row['reason']:
            current=dict(first_seconds=row['seconds'],last_seconds=row['seconds'],reason=row['reason'],frames=0);intervals.append(current)
        current['last_seconds']=row['seconds'];current['frames']+=1
    stalls=[]
    for before,after in zip(relays,relays[1:]):
        start,end=before['end_ms'],after['start_ms']
        if end-start<100:continue
        inside=[s for s in sampler if start<=s['time_ms']<=end]
        left=[s for s in sampler if s['time_ms']<=start];right=[s for s in sampler if s['time_ms']>=end]
        context=left[-1:]+inside+right[:1]
        stalls.append(dict(start_ms=start,end_ms=end,poll_gap_ms=end-start,
            sampler_records_inside=len(inside),observed_head_flags=sorted({s['head_flags'] for s in context}),
            largest_sampler_gap_ms=max([b['time_ms']-a['time_ms'] for a,b in zip(context,context[1:])],default=None)))
    return dict(session=session.name,counts=dict(counts),missing_intervals=intervals,relay_poll_stalls=stalls,
                stricter_audit_disagrees_with_existing_pose_join=differences),rows


def audit_mesh(session):
    from frame_data import Recording
    recording=Recording(session/'frame/sets.bin');raw={x[0] for x in recording.index}
    stamps=[];invalid=collections.Counter()
    with (session/'predictions.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);stamps.append(row['capture_raw_ns'])
            for key in ['capture_pose','current_pose']:
                if not row[key]['valid']:invalid[key]+=1
    if len(set(stamps))!=len(stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):
        raise ValueError('Mesh predictions not strictly ordered')
    selected=set(stamps)
    return dict(session=session.name,raw_frames=len(raw),processed_frames=len(stamps),
        invalid_pose_counts=dict(invalid),missing_raw_inside_processed_span=sum(stamps[0]<=t<=stamps[-1] and t not in selected for t in raw),
        capture_gaps_over_100ms=[dict(after_seconds=(a-recording.index[0][0])/1e9,gap_ms=(b-a)/1e6) for a,b in zip(stamps,stamps[1:]) if b-a>100_000_000])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,action='append',default=[])
    p.add_argument('--mesh',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);reports=[]
    for session in a.session:
        summary,rows=audit_session(session);reports.append(summary)
        (a.output/(session.name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(scope=__doc__,sessions=reports,mesh_sessions=[audit_mesh(s) for s in a.mesh],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(sessions=[dict(session=r['session'],counts=r['counts'],join_differences=len(r['stricter_audit_disagrees_with_existing_pose_join'])) for r in reports],mesh=[dict(session=r['session'],processed=r['processed_frames'],invalid=r['invalid_pose_counts'],gaps_over100ms=len(r['capture_gaps_over_100ms'])) for r in report['mesh_sessions']]),indent=2))


if __name__=='__main__':main()
