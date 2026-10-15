"""Offline raw-image palm-motion experiment. No tracking output or live access."""
import argparse
import collections
import hashlib
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from frame_data import Recording, read_audit
from palm_flow import PalmFlow, PARAMETERS

PALM = [0, 5, 9, 13, 17]


def crop_card(image, seed, result, title):
    points = np.asarray(result.get('palm', seed))
    combined = np.vstack((seed, points))
    lo = np.maximum(0, np.floor(combined.min(0)-65).astype(int))
    hi = np.minimum([image.shape[1], image.shape[0]], np.ceil(combined.max(0)+65).astype(int))
    shown = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(shown, [cv2.convexHull(np.rint(seed).astype(np.int32))], True, (160, 160, 160), 1)
    if result.get('supported'):
        cv2.polylines(shown, [cv2.convexHull(np.rint(points).astype(np.int32))], True, (0, 230, 255), 2)
        if 'region_hull' in result:
            cv2.polylines(shown, [np.rint(result['region_hull']).astype(np.int32)], True, (255, 200, 0), 1)
        for p in result['points']:
            cv2.circle(shown, tuple(np.rint(p).astype(int)), 2, (0, 255, 0), -1)
    crop = shown[lo[1]:hi[1], lo[0]:hi[0]]
    scale = min(2., 460/max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    card = np.zeros((530, 500, 3), np.uint8)
    card[50:50+crop.shape[0], 10:10+crop.shape[1]] = crop
    cv2.putText(card, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    cv2.putText(card, result['reason'], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return card


def run(case, folder):
    session = Path(case['session'])
    recording = Recording(session/'frame/sets.bin')
    first = recording.index[0][0]
    seed = None
    metadata = {}
    with Path(case['reference']).open() as source:
        for line in source:
            row = json.loads(line)
            seconds = (row['capture_raw_ns']-first)/1e9
            if seconds > case['start']+.8:
                break
            if seconds <= case['start']:
                seed = row
            if case['start']-.1 <= seconds:
                metadata[row['capture_raw_ns']] = dict(valid=row['capture_pose']['valid'],
                    baseline_present=any(h['side']==case['side'] for h in row['hands']))
    if seed is None or not seed['capture_pose']['valid']:
        raise ValueError('No valid seed capture')
    seed_seconds = (seed['capture_raw_ns']-first)/1e9
    if not 0 <= case['start']-seed_seconds < .05:
        raise ValueError('Seed is stale')
    hands = [h for h in seed['hands'] if h['side']==case['side']]
    if len(hands) != 1:
        raise ValueError('No unique seed anatomical slot')
    audit = read_audit(session/'frame/live-views.txt')
    sources = hands[0]['source_views']
    views = [v for v in audit[seed['capture_raw_ns']]['views'] if
             any((v['camera'], v['id'])==(s['camera'], s['source_id']) for s in sources)]
    if not views:
        raise ValueError('No exact seed source images')
    folder.mkdir()
    states = {}; outlines = {}; rows = []; cards = []; reasons = collections.Counter()
    timings = []; marked = set()
    for seconds, stamp, images in recording.frames(seed_seconds-.000001, seed_seconds+.8):
        if stamp == seed['capture_raw_ns']:
            for v in views:
                name = v['camera']; image = images[name][0]
                palm = v['points'][PALM].copy(); region = v['points'].copy()
                if case.get('control') == 'offset':
                    palm += np.asarray(case['offset']); region += np.asarray(case['offset'])
                if np.any(palm < 10) or np.any(palm >= np.array([image.shape[1]-10, image.shape[0]-10])):
                    raise ValueError('Seed/control palm outside image')
                states[name] = PalmFlow(image, palm, stamp, region if case.get('region')=='hand-hull' else None); outlines[name] = palm
                result = dict(reason=states[name].reason, supported=states[name].active,
                    palm=palm.tolist(), region_hull=states[name].region_hull.tolist(), points=states[name].points.reshape(-1,2).tolist())
                cards.append(crop_card(image, palm, result, f'{name} seed {seconds:.3f}s'))
            continue
        meta = metadata.get(stamp)
        if meta is None:
            raise ValueError('Missing exact reference context')
        started = time.perf_counter()
        for name, state in states.items():
            raw = images[name][0]
            image = np.zeros_like(raw) if case.get('control')=='zero_after_seed' else raw
            if not meta['valid']:
                state.active = False; state.reason = 'missing_capture_pose_evidence'
            result = state.advance(image, stamp)
            reasons[result['reason']] += 1
            rows.append(dict(seconds=seconds, stamp=stamp, camera=name,
                             baseline_present=meta['baseline_present'], **result))
            age = (stamp-seed['capture_raw_ns'])/1e9
            for target in [.2, .4, .6, .7]:
                key=(name,target)
                if age >= target and key not in marked:
                    cards.append(crop_card(image, outlines[name], result,
                        f'{name} {seconds:.3f}s control={case.get("control", "real")}'))
                    marked.add(key)
        timings.append((time.perf_counter()-started)*1000)
    # Timing above includes occasional review rendering; no live performance claim.
    (folder/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    if len(cards)%2:cards.append(np.zeros_like(cards[0]))
    cv2.imwrite(str(folder/'review.png'), np.vstack([np.hstack(cards[i:i+2]) for i in range(0,len(cards),2)]))
    summary = dict(case=case, seed_seconds=seed_seconds,
        seed_source_views=sources, seed_feature_counts={n:s.initial_count for n,s in states.items()},
        seed_mask_pixels={n:s.mask_pixels for n,s in states.items()},
        view_updates=len(rows), supported=sum(r['supported'] for r in rows),
        supported_during_baseline_absence=sum(r['supported'] and not r['baseline_present'] for r in rows),
        last_supported_seconds={n:max([r['seconds'] for r in rows if r['camera']==n and r['supported']],default=None) for n in states},
        reasons=dict(reasons), offline_ms_with_review_p50_p95_max=np.percentile(timings,[50,95,100]).tolist())
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def review_transitions(folder):
    """Render first/last supported and first failed updates from saved output.

    This does not rerun flow or choose a new seed. Failure can happen before the
    regular 200 ms review interval, so transition images are essential evidence.
    """
    report=json.loads((folder/'summary.json').read_text())
    rows=json.loads((folder/'rows.json').read_text());case=report['case']
    session=Path(case['session']);recording=Recording(session/'frame/sets.bin')
    audit=read_audit(session/'frame/live-views.txt')
    seed_stamp=min(audit,key=lambda t:abs((t-recording.index[0][0])/1e9-report['seed_seconds']))
    outlines={}
    for source in report['seed_source_views']:
        view=next(v for v in audit[seed_stamp]['views'] if
                  (v['camera'],v['id'])==(source['camera'],source['source_id']))
        palm=view['points'][PALM].copy()
        if case.get('control')=='offset':palm+=np.asarray(case['offset'])
        outlines[source['camera']]=palm
    wanted={}
    for name in outlines:
        selected=[r for r in rows if r['camera']==name]
        supported=[r for r in selected if r['supported']]
        failed=[r for r in selected if not r['supported']]
        for label,items in [('first supported',supported[:1]),('last supported',supported[-1:]),('first failure',failed[:1])]:
            for row in items:wanted.setdefault(row['stamp'],[]).append((label,row))
    cards=[]
    for seconds,stamp,images in recording.frames(report['seed_seconds'],report['seed_seconds']+.8):
        for label,row in wanted.get(stamp,[]):
            name=row['camera'];image=images[name][0]
            if case.get('control')=='zero_after_seed':image=np.zeros_like(image)
            cards.append(crop_card(image,outlines[name],row,f'{name} {seconds:.3f}s {label}'))
    if len(cards)%2:cards.append(np.zeros_like(cards[0]))
    cv2.imwrite(str(folder/'transitions.png'),np.vstack([np.hstack(cards[i:i+2]) for i in range(0,len(cards),2)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ['probe_palm_flow.py','palm_flow.py','frame_data.py']:
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    shutil.copy2(a.protocol,a.output/'protocol.json')
    reports=[]
    for case in protocol['cases']:
        try:
            result=run(case,a.output/case['name'])
            review_transitions(a.output/case['name'])
        except ValueError as exc:
            result=dict(case=case,error=str(exc))
        reports.append(result)
        print(json.dumps({k:result[k] for k in ['case','error','supported','supported_during_baseline_absence','reasons'] if k in result}),flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(region_variant='palm-interior: 5-point palm hull eroded 5px; optional hand-hull: 21-point hand hull eroded 1px. All remaining thresholds unchanged.',scope=__doc__+' Support means coherent image texture only, never confirmed physical hand presence, anatomical identity or 3D pose.',parameters=PARAMETERS,opencv=cv2.__version__,opencv_threads=1,protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},runs=reports),indent=2)+'\n')
    if any('error' in r for r in reports):raise SystemExit(2)


if __name__=='__main__':main()
