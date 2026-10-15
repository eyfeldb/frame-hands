import unittest
import torch
from target_chirality import condition


class TargetMapTests(unittest.TestCase):
    def test_horizontal_mirror_preserves_conditioning(self):
        image=torch.arange(96*96,dtype=torch.float32).reshape(1,1,96,96)/9216
        points=torch.arange(40,dtype=torch.float32).reshape(1,20,2)+12.25
        mirrored=points.clone();mirrored[:,:,0]=95-mirrored[:,:,0]
        self.assertTrue(torch.allclose(condition(image,points).flip(-1),condition(image.flip(-1),mirrored),atol=1e-6))

    def test_wrist_target_matches_pixel_center(self):
        image=torch.zeros(1,1,96,96);points=torch.ones(1,20,2)*30;points[:,0]=torch.tensor([20.,40.])
        maps=condition(image,points)
        self.assertEqual(maps.shape,(1,3,96,96))
        self.assertEqual(float(maps[0,1,40,20]),1.)
        self.assertEqual(float(maps[0,2,30,30]),1.)
        points[0,0,0]=float('nan')
        with self.assertRaises(ValueError):condition(image,points)


if __name__=='__main__':unittest.main()
