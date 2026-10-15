import json,tempfile,unittest
from pathlib import Path
from compare_native import compare


class NativeComparisonTests(unittest.TestCase):
    def run_pair(self,left,right):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,row in (('left',left),('right',right)):(root/name).write_text(json.dumps(row)+'\n')
            return compare(root/'left',root/'right')

    def test_different_source_views_are_not_same_identity(self):
        def hand(source):return dict(side=0,source_id=1,source_views=[dict(camera='left',source_id=source)])
        result=self.run_pair(dict(stamp=1,hands=[hand(2)]),dict(stamp=1,hands=[hand(3)]))
        self.assertEqual(result['assignment_changed_frames'],1)
        self.assertEqual(result['availability_changed_frames'],0)

    def test_unaligned_captures_rejected(self):
        with self.assertRaises(ValueError):self.run_pair(dict(stamp=1,hands=[]),dict(stamp=2,hands=[]))

    def test_numerical_error_and_disappearance(self):
        def hand(x):return dict(side=1,source_views=[dict(camera='right',source_id=3)],vertices_world_mm=[[x,0,0]],wrist_world_mm=[[1,0,0,0]],joint_angles=[0])
        result=self.run_pair(dict(stamp=1,hands=[hand(0)]),dict(stamp=1,hands=[hand(.005)]))
        self.assertEqual(result['assignment_changed_frames'],0)
        self.assertEqual(result['max_vertex_coordinate_difference_mm_p50_p95_max'],[.005]*3)
        result=self.run_pair(dict(stamp=1,hands=[hand(0)]),dict(stamp=1,hands=[]))
        self.assertEqual(result['availability_changed_frames'],1)


if __name__=='__main__':unittest.main()
