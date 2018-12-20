# vim: expandtab:ts=4:sw=4
from .track import Track

class Track_RS(Track):
    """
    See Track for proper documentation. Track_RS mostly leverages on the functionality of Track and on top of that stores the detected classes.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, det_class=None):
        super().__init__(mean, covariance, track_id, n_init, max_age,
                 feature)

        self.det_classes = []
        if det_class is not None:
            self.det_classes.append(det_class)

    def update(self, kf, detection):
        super().update(kf, detection)
        self.det_classes.append(detection.det_class)
