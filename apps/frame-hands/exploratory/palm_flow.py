"""Short-lived 2D palm-region motion evidence, not hand presence or pose output.

Existing detector pixels initialize a track once. No later detections, models,
identity labels or feature replenishment repair it. A textured background can
pass these checks: callers must not treat continuation as proof of a hand.
"""
import cv2
import numpy as np

PARAMETERS = dict(max_age_ms=650, max_gap_ms=100, min_points=8,
                  max_corners=80, quality=.01, min_distance=4,
                  erosion_px=5, fb_px=1., photometric_l1=20.,
                  max_step_px=80., retained_fraction=.5,
                  ransac_px=2., inlier_fraction=.7,
                  scale_per_step=[.8, 1.25])
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))


def validate_image(image):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected grayscale uint8 image')


class PalmFlow:
    def __init__(self, image, palm, stamp, region_points=None):
        validate_image(image)
        palm = np.asarray(palm, np.float32)
        if palm.shape != (5, 2) or not np.isfinite(palm).all():
            raise ValueError('Expected finite wrist and four knuckles')
        self.seed_stamp = self.stamp = int(stamp)
        self.image = image.copy()
        self.palm = palm.copy()
        mask = np.zeros_like(image)
        region = palm if region_points is None else np.asarray(region_points, np.float32)
        if region_points is not None and (region.shape != (21, 2) or not np.isfinite(region).all()):
            raise ValueError('Hand-region variant requires 21 finite detector points')
        hull = cv2.convexHull(region).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 255)
        kernel = 11 if region_points is None else 3
        mask = cv2.erode(mask, np.ones((kernel, kernel), np.uint8))
        self.mask_pixels = int(np.count_nonzero(mask))
        self.region = 'palm-interior' if region_points is None else 'hand-hull'
        self.region_hull = cv2.convexHull(region).reshape(-1, 2)
        points = cv2.goodFeaturesToTrack(image, maxCorners=80, qualityLevel=.01,
                                       minDistance=4, mask=mask, blockSize=7)
        self.points = points if points is not None else np.empty((0, 1, 2), np.float32)
        self.initial_count = len(self.points)
        self.reason = 'seeded' if self.initial_count >= 8 else 'insufficient_seed_texture'
        self.active = self.initial_count >= 8

    def advance(self, image, stamp):
        validate_image(image)
        stamp = int(stamp)
        info = dict(seed_points=self.initial_count, input_points=len(self.points),
                    age_ms=(stamp-self.seed_stamp)/1e6, hypothesis_only=True)
        def fail(reason):
            self.active = False
            self.reason = reason
            return dict(info, supported=False, reason=reason)
        if not self.active:
            return fail('inactive_' + self.reason.removeprefix('inactive_'))
        if image.shape != self.image.shape:
            return fail('image_shape_changed')
        if not 0 < stamp-self.stamp <= 100_000_000:
            return fail('invalid_capture_gap')
        if stamp-self.seed_stamp > 650_000_000:
            return fail('seed_expired')
        p1, status, error = cv2.calcOpticalFlowPyrLK(self.image, image, self.points, None, **LK)
        if p1 is None:
            return fail('no_forward_flow')
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(image, self.image, p1, None, **LK)
        if back is None:
            return fail('no_backward_flow')
        old = self.points.reshape(-1, 2)
        new = p1.reshape(-1, 2)
        fb = np.linalg.norm(back.reshape(-1, 2)-old, axis=1)
        height, width = image.shape
        valid = (status.ravel() != 0) & (backward_status.ravel() != 0)
        valid &= np.isfinite(new).all(1) & np.isfinite(fb) & (fb <= 1.)
        valid &= np.isfinite(error.ravel()) & (error.ravel() <= 20.)
        valid &= np.linalg.norm(new-old, axis=1) <= 80.
        valid &= (new[:, 0] >= 10) & (new[:, 0] < width-10)
        valid &= (new[:, 1] >= 10) & (new[:, 1] < height-10)
        info['consistent_points'] = int(valid.sum())
        if valid.sum() < max(8, self.initial_count*.5):
            return fail('insufficient_consistent_texture')
        cv2.setRNGSeed(7421)
        matrix, inliers = cv2.estimateAffinePartial2D(old[valid], new[valid],
            method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=1000,
            confidence=.99, refineIters=10)
        if matrix is None or not np.isfinite(matrix).all():
            return fail('no_coherent_motion')
        keep = inliers.ravel().astype(bool)
        info['inliers'] = int(keep.sum())
        info['inlier_fraction'] = float(keep.mean())
        scale = float(np.linalg.norm(matrix[0, :2]))
        info['scale'] = scale
        if keep.sum() < max(8, self.initial_count*.5) or keep.mean() < .7:
            return fail('incoherent_motion')
        if not .8 <= scale <= 1.25:
            return fail('excessive_scale_change')
        moved = self.palm @ matrix[:, :2].T + matrix[:, 2]
        if not np.isfinite(moved).all():
            return fail('invalid_palm_motion')
        self.region_hull = (self.region_hull @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)
        self.palm = moved.astype(np.float32)
        self.points = new[valid][keep].reshape(-1, 1, 2).copy()
        self.image = image.copy()
        self.stamp = stamp
        self.reason = 'image_motion_supported'
        return dict(info, supported=True, reason=self.reason,
                    palm=self.palm.tolist(), region_hull=self.region_hull.tolist(), points=self.points.reshape(-1, 2).tolist(),
                    fb_px_median=float(np.median(fb[valid][keep])),
                    affine=matrix.tolist())
