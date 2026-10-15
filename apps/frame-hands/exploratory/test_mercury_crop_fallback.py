import unittest
from replay_mercury_crop_fallback import choose_geometry_views


class CropFallbackTests(unittest.TestCase):
    def test_low_presence_in_original_crop_never_triggers_alternative(self):
        low=dict(active=False,presence=.1);high=dict(active=True,presence=1.)
        observations=dict(box=dict(slam_left=low,slam_right=high),rays=dict(slam_left=high,slam_right=low))
        views,choices=choose_geometry_views(observations)
        self.assertIs(views['slam_left'],low)
        self.assertEqual(choices,dict(slam_left='box',slam_right='box'))
    def test_missing_geometry_uses_only_available_view_without_mutating_sources(self):
        left=dict(active=True);right=dict(active=True)
        observations=dict(box=dict(slam_left=left),rays=dict(slam_right=right))
        views,choices=choose_geometry_views(observations)
        self.assertIs(views['slam_left'],left);self.assertIs(views['slam_right'],right)
        self.assertEqual(choices,dict(slam_left='box',slam_right='rays'))
        self.assertNotIn('slam_right',observations['box'])
        views,choices=choose_geometry_views(dict(box=dict(slam_left=left)))
        self.assertEqual(set(views),{'slam_left'});self.assertEqual(choices,dict(slam_left='box'))
    def test_missing_or_inactive_alternative_cannot_manufacture_active_pair(self):
        self.assertEqual(choose_geometry_views({}),({},{}))
        inactive=dict(active=False)
        views,choices=choose_geometry_views(dict(rays=dict(slam_right=inactive)))
        self.assertIs(views['slam_right'],inactive)
        self.assertFalse(views['slam_right']['active'])
        self.assertEqual(choices,dict(slam_right='rays'))


if __name__=='__main__':unittest.main()
