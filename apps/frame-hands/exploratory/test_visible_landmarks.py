import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from score_visible_landmarks import match_regions, validate_annotations


def label(i=0, tips=None, xy=(10., 10.)):
    return dict(label_id=i, xy=list(xy), radius_px=2.,
                possible_native_tips=[0] if tips is None else tips,
                visibility='visible distal region', stamp=123,
                camera='left', crop_source_id=7, physical_side=0)


class RegionMatchingTests(unittest.TestCase):
    def test_ambiguous_regions_cannot_reuse_the_same_finger(self):
        tips = np.full((5, 2), np.nan)
        tips[1] = [10., 10.]
        tips[2] = [11., 10.]
        rows = match_regions([label(0, [1, 2]), label(1, [1])], tips)
        self.assertEqual([r['matched_tip'] for r in rows], [2, 1])
        self.assertTrue(all(r['status'] == 'within_region' for r in rows))

    def test_anatomically_wrong_but_nearby_tip_cannot_count(self):
        tips = np.full((5, 2), np.nan)
        tips[0] = [20., 10.]
        tips[1] = [10., 10.]
        result = match_regions([label()], tips)[0]
        self.assertEqual(result['matched_tip'], 0)
        self.assertEqual(result['outside_radius_px'], 8.)

    def test_missing_projection_is_not_a_success_or_a_finite_error(self):
        result = match_regions([label()], np.full((5, 2), np.nan))[0]
        self.assertEqual(result['status'], 'unmatched_tip')
        self.assertIsNone(result['distance_px'])
        json.dumps(result, allow_nan=False)

    def test_boundary_inside_and_occluded_claim_rejected(self):
        tips = np.full((5, 2), np.nan)
        tips[0] = [12., 10.]
        self.assertEqual(match_regions([label()], tips)[0]['status'], 'within_region')
        row = label()
        row['visibility'] = 'occluded'
        with self.assertRaises(ValueError):
            match_regions([row], tips)

    def test_duplicate_or_nonfinite_claims_rejected(self):
        tips = np.zeros((5, 2))
        for rows in ([label(), label()], [label(tips=[0, 0])],
                     [label(xy=(float('nan'), 0.))]):
            with self.assertRaises(ValueError):
                match_regions(rows, tips)


class AnnotationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.raw = self.root / 'left.png'
        self.raw.write_bytes(b'exact saved pixels')
        digest = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        row = label()
        row['raw_sha256'] = digest
        self.data = dict(frozen_before_candidate_projection=True, labels=[row])
        self.manifest = dict(frames=[dict(stamp=123, crops=[dict(
            camera='left', source_id=7, path='left-source7.png', raw_sha256=digest)])])

    def validate(self, review=None):
        return validate_annotations(self.data, self.manifest, self.root, 'label-digest', review)

    def test_rejection_preserves_claim_and_requires_matching_file(self):
        before = copy.deepcopy(self.data)
        review = dict(labels_sha256='label-digest', rejected_labels=[
            dict(label_id=0, reason='Anatomical identity is uncertain')])
        self.assertEqual(self.validate(review), {0: 'Anatomical identity is uncertain'})
        self.assertEqual(self.data, before)
        review['labels_sha256'] = 'another-file'
        with self.assertRaises(ValueError):
            self.validate(review)

    def test_annotation_hash_and_actual_pixels_both_checked(self):
        self.assertEqual(self.validate(), {})
        self.data['labels'][0]['raw_sha256'] = 'wrong-annotation-reference'
        with self.assertRaises(ValueError):
            self.validate()
        self.data['labels'][0]['raw_sha256'] = self.manifest['frames'][0]['crops'][0]['raw_sha256']
        self.raw.write_bytes(b'different pixels')
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_global_id_fails_even_across_anatomical_sides(self):
        second = dict(self.data['labels'][0], physical_side=1)
        self.data['labels'].append(second)
        with self.assertRaises(ValueError):
            self.validate()

    def test_exact_capture_and_unique_crop_required(self):
        self.data['labels'][0]['stamp'] += 1
        with self.assertRaises(ValueError):
            self.validate()
        self.data['labels'][0]['stamp'] -= 1
        self.manifest['frames'][0]['crops'] *= 2
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
