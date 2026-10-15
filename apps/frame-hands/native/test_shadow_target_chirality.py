import unittest
from shadow_target_chirality import shortcut


class ShortcutTests(unittest.TestCase):
    def pair(self):return dict(camera='left',id=1,right=.9),dict(camera='right',id=1,right=.9)
    def test_both_cameras_and_mirrors_must_agree(self):
        a,b=self.pair();scores={('left',1):(.99,.01),('right',1):(.99,.01)}
        self.assertEqual(shortcut(a,b,scores),(.98,'shortcut'))
        scores['right',1]=(.01,.99);b['right']=.1
        self.assertEqual(shortcut(a,b,scores),(None,'camera_conflict'))
        scores['right',1]=(.99,.99)
        self.assertEqual(shortcut(a,b,scores),(None,'weak_or_mirror_conflict'))
    def test_raw_detector_cannot_be_overridden(self):
        a,b=self.pair();a['right']=.2;scores={('left',1):(.99,.01),('right',1):(.99,.01)}
        self.assertEqual(shortcut(a,b,scores),(None,'detector_conflict'))
    def test_nonfinite_and_out_of_range_rejected(self):
        a,b=self.pair()
        for q in (float('nan'),1.2,-.2):
            scores={('left',1):(q,.01),('right',1):(.99,.01)}
            self.assertEqual(shortcut(a,b,scores),(None,'invalid'))


if __name__=='__main__':unittest.main()
