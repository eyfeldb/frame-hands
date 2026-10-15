import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import calibrate_gaze as cal


TARGETS=[(0,0),(-16,10),(16,-10),(-16,-10),(16,10),(0,-10),(-16,0),(0,10),(16,0),(-9,6),(9,-6),(-9,-6),(9,6),(3,2)]


def recording(kind='offset'):
    rows=[dict(session='synthetic-session',created=1000)]
    for i,(x,y) in enumerate(TARGETS):
        if kind=='offset':target=(x+3,y-2)
        elif kind=='affine':target=(x*1.12+y*.08+3,y*.85+x*.04-2)
        else:target=(x+3+.008*y*y,y-2+.007*x*x+.01*x*y)
        samples=[[x+.08*math.sin(j),y+.08*math.cos(j),*target] for j in range(70)]
        rows.append(dict(point=i,validation=i>=9,total_polls=75,valid_polls=70,samples=samples))
    return rows+[dict(completed=True)]


class CalibrationTests(unittest.TestCase):
    def test_offset_affine_quadratic_and_holdout_accuracy(self):
        for kind,terms in [('offset',1),('affine',3),('quadratic',6)]:
            report=cal.analyze(recording(kind))
            self.assertTrue(report['accepted'],report)
            self.assertEqual(report['terms'],terms)
            self.assertLess(report['corrected_median_deg'],.15)

    def test_validation_is_not_training_or_model_selection(self):
        rows=recording('quadratic');before=cal.analyze(rows)
        for r in rows[10:-1]:
            for s in r['samples']:s[2]+=7
        after=cal.analyze(rows)
        self.assertFalse(after['accepted'])
        self.assertEqual(before['coefficient'],after['coefficient'])
        self.assertEqual(before['terms'],after['terms'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gaze.profile'
            with self.assertRaises(ValueError):cal.write_profile(after,path)
            self.assertFalse(path.exists())

    def test_missing_invalid_unstable_stuck_data(self):
        for variant in ('missing','nan','jitter','stuck','partition'):
            rows=recording()
            if variant=='missing':rows.pop(3)
            if variant=='nan':rows[1]['samples'][3][0]=float('nan')
            if variant=='jitter':
                for j,s in enumerate(rows[1]['samples']):s[0]+=5*(-1)**j
            if variant=='stuck':
                for r in rows[1:-1]:
                    for s in r['samples']:s[0]=s[1]=0
            if variant=='partition':rows[2]['validation']=True
            with self.assertRaises(ValueError,msg=variant):cal.analyze(rows)

    def test_native_profile_matches_fit(self):
        report=cal.analyze(recording('quadratic'))
        binary=Path(__file__).parent/'build/test-gaze-math'
        if not binary.exists():self.skipTest('Native reader not built')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gaze.profile';cal.write_profile(report,path)
            actual=json.loads(subprocess.check_output([str(binary),str(path),'synthetic-session','1010'],text=True))
            expected=cal.correct([4,-3],report['coefficient'])
            for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=6)

    def test_refinement_frozen_before_new_validation_and_preview_is_not_active(self):
        rows=recording('quadratic')
        training=[cal.summarize(r) for r in rows[1:10]]
        fixed=dict(terms=6,coefficient=cal.fit(training,6,.5))
        first=cal.analyze(rows,fixed=fixed)
        for r in rows[10:-1]:
            for s in r['samples']:s[2]+=6
        second=cal.analyze(rows,fixed=fixed)
        self.assertEqual(first['coefficient'],second['coefficient'])
        self.assertFalse(second['accepted'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.preview'
            cal.write_profile(second,path,preview=True)
            self.assertTrue(path.read_text().startswith('FHGAZE_PREVIEW1\n'))


if __name__=='__main__':unittest.main()
