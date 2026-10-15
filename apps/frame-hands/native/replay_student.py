"""Causal paired replay of teacher versus a distilled encoder; saved crops only."""
import argparse, io, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exploratory'))
from frame_data import read_audit, load_calibration
from replay_mesh_trial import packets
from run_mesh_trial import pose_matrix
from identity_pipeline import ImageIdentityPipeline
from student_encoder import load_student
from distill_encoder import sha
from compare_student import hand_key


def quantiles(values):
    return np.percentile(values, [50, 95, 100]).tolist() if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'session', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    pipelines = [ImageIdentityPipeline(a.upstream), ImageIdentityPipeline(a.upstream)]
    student = load_student(a.upstream, a.checkpoint)
    candidate = pipelines[1]
    for model in (candidate.evidence.backend.model, candidate.backend.model):
        model._feature_extractor._image_backbone = student
    candidate.evidence.regressor.warmup()
    candidate.backend.prepared_regressor.warmup()
    for pipeline in pipelines:
        pipeline.bind_calibration(load_calibration(a.session / 'calibration'))
    context = {}
    for line in (a.session / 'predictions.jsonl').open():
        row = json.loads(line)
        context[row['capture_raw_ns']] = {k: row[k] for k in ('capture_mono_ns', 'capture_pose', 'current_pose')}
    del row
    frames = matched = agreement_changes = view_pairs = assignment_changes = 0
    counts = [0, 0]
    errors, maxima, probabilities = [], [], []
    times = [[], []]
    with (a.output / 'teacher.jsonl').open('w') as tf, (a.output / 'student.jsonl').open('w') as sf, (a.output / 'comparison.jsonl').open('w') as cf:
        for meta, blobs in packets(a.session / 'transport.bin'):
            if meta.get('waiting'):
                continue
            reference = context.get(meta['stamp'])
            if reference is None:
                for pipeline in pipelines:
                    pipeline.reset('Missing recorded capture context')
                continue
            observation = next(iter(read_audit(io.StringIO(meta['audit'])).values()))
            prepared = {(v['camera'], v['id']): (cv2.imdecode(np.frombuffer(b, np.uint8), 0), v) for v, b in zip(meta['crops'], blobs)}
            world = pose_matrix(reference['capture_pose']['capture']) if reference['capture_pose']['valid'] else None
            results = []
            for n, (pipeline, out) in enumerate(zip(pipelines, (tf, sf))):
                start = time.perf_counter()
                result = pipeline.predict(observation, world, prepared)
                times[n].append((time.perf_counter() - start) * 1000)
                if not reference['current_pose']['valid']:
                    result['hands'] = []
                counts[n] += len(result['hands'])
                results.append(result)
                out.write(json.dumps(dict(capture_raw_ns=meta['stamp'], **reference, **result), allow_nan=False) + '\n')
            expected, actual = [{hand_key(h): h for h in r['hands']} for r in results]
            changed = expected.keys() != actual.keys()
            assignment_changes += changed
            row_errors = []
            for key in expected.keys() & actual.keys():
                delta = np.asarray(actual[key]['vertices_world_mm']) - expected[key]['vertices_world_mm']
                error = float(np.linalg.norm(delta, axis=-1).mean())
                maximum = float(np.abs(delta).max())
                errors.append(error); maxima.append(maximum); matched += 1
                row_errors.append(dict(side=key[0], source_views=key[1], mean_vertex_distance_mm=error, max_coordinate_error_mm=maximum))
            evidence = [{(v['camera'], v['source_id']): v for v in p.last_debug.get('views', [])} for p in pipelines]
            if evidence[0].keys() != evidence[1].keys():
                raise ValueError('Different input evidence views')
            for key, value in evidence[0].items():
                other = evidence[1][key]
                view_pairs += 1
                agreement_changes += value['agreement'] != other['agreement']
                probabilities.append(abs(value['right'] - other['right']))
            cf.write(json.dumps(dict(stamp=meta['stamp'], assignment_changed=changed, teacher_keys=list(expected), student_keys=list(actual), matched=row_errors), allow_nan=False) + '\n')
            frames += 1
            if frames % 100 == 0:
                print(frames, 'frames', flush=True)
    result = dict(scope='Teacher imitation on exact recorded inputs. Differences are not physical ground-truth errors. No native/live speed claim.',
        recording=a.session.name, checkpoint_sha256=sha(a.checkpoint), frames=frames, teacher_hands=counts[0], student_hands=counts[1],
        assignment_changed_frames=assignment_changes, matched_hands=matched, matched_mean_vertex_distance_mm_p50_p95_max=quantiles(errors),
        matched_max_coordinate_error_mm_p50_p95_max=quantiles(maxima), evidence_view_pairs=view_pairs, agreement_changed_views=agreement_changes,
        evidence_right_score_difference_p50_p95_max=quantiles(probabilities), teacher_processing_ms_p50_p95_max=quantiles(times[0]), student_processing_ms_p50_p95_max=quantiles(times[1]),
        torch=torch.__version__, cuda=torch.version.cuda, tf32=False)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
