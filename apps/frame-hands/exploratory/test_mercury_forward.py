from pathlib import Path
import unittest
import numpy as np
from mercury_forward import NativeHandGenerator

DLL=Path(__file__).resolve().parents[3]/'.local/hand-alternatives/mercury-solver-forward-v1/mercury_solver.dll'


@unittest.skipUnless(DLL.exists(),'Optional native forward generator unavailable')
class ForwardGeneratorTests(unittest.TestCase):
    def test_translation_reflection_and_quaternions(self):
        gen=NativeHandGenerator(DLL);angles=np.tile([-.3,-.6,-.4],(4,1));wrist=np.array([0.,0.,0.,1.,0.,0.,-.5])
        left=gen.generate(wrist,angles,0);right=gen.generate(wrist,angles,1)
        reflected=left[:,:3].copy();reflected[:,0]*=-1;np.testing.assert_allclose(reflected,right[:,:3],atol=1e-7)
        moved=wrist.copy();moved[4:]+=[.02,-.04,.01]
        translated=gen.generate(moved,angles,0)
        np.testing.assert_allclose(translated[:,:3]-left[:,:3],np.tile([.02,-.04,.01],(26,1)),atol=6e-8)
        np.testing.assert_allclose(np.linalg.norm(left[:,3:],axis=1),1,atol=2e-6)
    def test_bone_lengths_survive_articulation_and_scale(self):
        gen=NativeHandGenerator(DLL);wrist=[.70710678,0,0,.70710678,0,0,-.5]
        edges=[(i,i+1) for lo,hi in [(2,5),(6,10),(11,15),(16,20),(21,25)] for i in range(lo,hi)]
        def lengths(x):return [np.linalg.norm(x[i,:3]-x[j,:3]) for i,j in edges]
        for side in (0,1):
            opened=gen.generate(wrist,np.tile([-.1,-.05,-.05],(4,1)),side)
            closed=gen.generate(wrist,np.tile([-1.2,-1.2,-.8],(4,1)),side)
            scaled=gen.generate(wrist,np.tile([-1.2,-1.2,-.8],(4,1)),side,size=.108)
            np.testing.assert_allclose(lengths(opened),lengths(closed),atol=1e-7)
            np.testing.assert_allclose(np.asarray(lengths(closed))*1.2,lengths(scaled),atol=1e-7)
    def test_invalid_inputs_do_not_generate_truth(self):
        gen=NativeHandGenerator(DLL);w=[0,0,0,1,0,0,-.5];a=np.zeros((4,3))
        for wrist,angles,side,size in [(w,np.ones((4,3)),0,.09),(w,a,2,.09),(w,a,0,.2),([0]*7,a,0,.09),(w,np.full((4,3),np.nan),0,.09),(w,np.zeros((3,3)),0,.09)]:
            with self.assertRaises(ValueError):gen.generate(wrist,angles,side,size)


if __name__=='__main__':unittest.main()
