# Deep SORT

## Introduction

A more realtime adaptation of Deep SORT.
Adapted from the official repo of *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT): [Git Repo](https://github.com/nwojke/deep_sort)
See the Deep Sort's paper [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.

## Dependencies

- Python 3
- NumPy
- Scipy
- cv2
- Embedder requires Pytorch & Torchvision

## Run

Example usage:
- Include this as submodule

```python
from deep_sort.deepsort_tracker_emb import DeepSort
tracker = DeepSort(max_age=30, nn_budget=70)
bbs = object_detector.detect(frame)
tracks = trackers.update_tracks(frame, bbs)
for track in tracks:
   ltrb = track.to_ltrb()
```

## Differences from original repo

- remove "academic style" offline processing style and implemented it to take in real-time detections and output accordingly.
- provides both options of using an in-built appearance feature embedder or to provide embeddings during update
- added (pytorch) mobilenetv2 as embedder (torch ftw).
- skip nms completely in preprocessing detections if nms_max_overlap == 1.0 (which is the default), in the original repo, nms will still be done even if threshold is set to 1.0 (probably because it was not optimised for speed).
- deepsort takes in a "clock" object (see `utils/clock.py` for example), which provides date for track naming and facilities track id reset every day, preventing overflow and overly large track ids when system runs for a long time.
- other minor adjustments.

## [From original repo] Highlevel overview of source files in `deep_sort`

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.
