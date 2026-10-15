import json
from pathlib import Path
import tempfile
import unittest
from compare_student import summarize


def hand(track=1001, view=7, side=0):
    return dict(side=side, source_id=track, source_views=[dict(camera='slam_left', source_id=view)],
        vertices_world_mm=[[0, 0, 0]], wrist_world_mm=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], joint_angles=[0])


class CrossRunComparisonTests(unittest.TestCase):
    def compare(self, reference, candidate):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, h in (('teacher', reference), ('student', candidate)):
                (root / (label + '.jsonl')).write_text(json.dumps(dict(capture_raw_ns=1, hands=[h])) + '\n')
            return summarize(root)

    def test_different_generation_counters_are_not_assignment_changes(self):
        result = self.compare(hand(track=1001), hand(track=1080))
        self.assertEqual(result['assignment_changed_frames'], 0)
        self.assertEqual(result['matched_hands'], 1)
        self.assertEqual(result['matched_mean_vertex_distance_mm_p50_p95_max'], [0, 0, 0])

    def test_equal_counters_do_not_hide_different_camera_detections(self):
        result = self.compare(hand(view=7), hand(view=8))
        self.assertEqual(result['assignment_changed_frames'], 1)
        self.assertEqual(result['side_availability_changed_frames'], 0)
        self.assertEqual(result['matched_hands'], 0)

    def test_side_swap_is_not_compared_as_matching_pose(self):
        result = self.compare(hand(side=0), hand(side=1))
        self.assertEqual(result['assignment_changed_frames'], 1)
        self.assertEqual(result['side_availability_changed_frames'], 1)
        self.assertEqual(result['matched_hands'], 0)


if __name__ == '__main__':
    unittest.main()
