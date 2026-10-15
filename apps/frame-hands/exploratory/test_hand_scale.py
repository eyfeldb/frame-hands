import unittest
import numpy as np
import test_association as fixtures
from audit_hand_scale import triangulate_palms,scale_ratios,select_views,PALM_DETECTOR

class HandScaleTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AssociationTests();self.fixture.setUp();self.cameras=self.fixture.cameras
        self.points=np.array([[-10,0,350],[-35,60,348],[-15,65,351],[10,62,349],[30,50,350.]])
    def views(self,points):
        result=[]
        for n,c in self.cameras.items():
            uv=np.zeros((21,2));uv[PALM_DETECTOR]=c.eye_to_window(c.world_to_eye(points))
            result.append(dict(camera=n,id=1,presence=.99,right=.01,points=uv))
        return result
    def test_calibrated_geometry_recovers_known_size(self):
        for scale in (.75,1.,1.2):
            center=self.points.mean(0);truth=(self.points-center)*scale+center
            result,reason=triangulate_palms(self.cameras,self.views(truth));self.assertIsNotNone(result,reason)
            np.testing.assert_allclose(result['points_mm'],truth,atol=1e-7)
            measured=scale_ratios(self.points,result['points_mm'])
            self.assertAlmostEqual(measured['least_squares_scale'],scale,places=8)
            self.assertLess(measured['relative_shape_residual'],1e-8)
    def test_bad_stereo_and_monocular_refused(self):
        views=self.views(self.points)
        self.assertIsNone(triangulate_palms(self.cameras,views[:1])[0])
        views[1]['points'][PALM_DETECTOR,1]+=30
        self.assertIsNone(triangulate_palms(self.cameras,views)[0])
    def test_low_parallax_refused(self):
        points=self.points.copy();points[:,2]=1800
        self.assertIsNone(triangulate_palms(self.cameras,self.views(points))[0])
    def test_identity_and_overlap_quality_filters(self):
        views=self.views(self.points);hand=dict(side=0,source_views=[dict(camera=v['camera'],source_id=1) for v in views])
        self.assertIsNotNone(select_views(hand,dict(views=views))[0])
        views[0]['right']=.5;self.assertIsNone(select_views(hand,dict(views=views))[0]);views[0]['right']=.01
        extra=dict(views[0],id=2);self.assertIsNone(select_views(hand,dict(views=views+[extra]))[0])
    def test_similarity_recovers_rigid_scale_without_reflection(self):
        from audit_wrist_correspondence import similarity
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_rotvec([.3,-.4,.2]).as_matrix();translation=np.array([20.,-10.,40.])
        target=1.07*self.points@rotation+translation
        scale,r,t,rms=similarity(self.points,target)
        self.assertAlmostEqual(scale,1.07,places=10);self.assertLess(rms,1e-10)
        np.testing.assert_allclose(r,rotation,atol=1e-10);np.testing.assert_allclose(t,translation,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_native_scale_cpu_matches_saved_cuda_and_ignores_prior_crop_memory(self):
        import json,sys
        from pathlib import Path
        from frame_data import Recording,read_audit,load_calibration
        from live_backend import Backend
        from audit_hand_scale import learned_scale
        root=Path(__file__).resolve().parents[3];upstream=root/'.local/hand-alternatives/UmeTrack'
        session=root/'.local/hand-alternatives/results/live-mesh-20261005T050930Z'
        rows=[json.loads(s) for s in (root/'.local/hand-alternatives/results/hand-scale-audit-v1-latest/rows.jsonl').read_text().splitlines()]
        selected=[next(r for r in rows if r['side']==side and r['learned_scale'] is not None) for side in (0,1)]
        sys.path.insert(0,str(upstream));backend=Backend(upstream,load_calibration(session/'calibration'))
        audit=read_audit(session/'frame/live-views.txt');recording=Recording(session/'frame/sets.bin')
        def predict(r):
            views=[next(v for v in audit[r['stamp']]['views'] if (v['camera'],v['id'])==(key['camera'],key['source_id'])) for key in r['source_views']]
            raw=next(recording.frames(r['seconds']-.00001,r['seconds']+.00001));self.assertEqual(raw[1],r['stamp'])
            return learned_scale(backend,r['side'],views,raw[2])
        first=predict(selected[0]);second=predict(selected[1]);repeat=predict(selected[0])
        self.assertAlmostEqual(first,selected[0]['learned_scale'],places=5)
        self.assertAlmostEqual(second,selected[1]['learned_scale'],places=5)
        self.assertEqual(first,repeat)

    def test_nonuniform_shape_not_disguised_as_perfect_scale(self):
        changed=self.points.copy();changed[4,0]+=15
        self.assertGreater(scale_ratios(self.points,changed)['relative_shape_residual'],.05)

if __name__=='__main__':unittest.main()
