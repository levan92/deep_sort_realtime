# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from .track_RS import Track_RS
from .tracker import Tracker

class Tracker_RS(Tracker):
    """
    Multi-target tracker that stores the detected class. See Tracker for documentation.
    """
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        super().__init__(metric, max_iou_distance, max_age, n_init)

    # Overriding this method because we need to create Track_RS object instead of regular Track objects
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track_RS(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.det_class))
        self._next_id += 1
