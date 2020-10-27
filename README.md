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

```
from deep_sort.deepsort_tracker_emb import DeepSort
tracker = DeepSort(max_age=30, nn_budget=70)
bbs = object_detector.detect(frame)
tracks = trackers.update_tracks(frame, bbs)
for track in tracks:
   ltrb = track.to_ltrb()
```

## Parameters

For `deepsort_tracker_emb.py` (comes with in-built embedder):
```
Parameters
----------
max_age : Optional[int] = 30
   Maximum number of missed misses before a track is deleted.
nms_max_overlap : Optional[float] = 1.0
   Non-maxima suppression threshold: Maximum detection overlap
max_cosine_distance : Optional[float] = 0.2
   Gating threshold for cosine distance
nn_budget :  Optional[int] = None
   Maximum size of the appearance descriptors, if None, no budget is enforced
override_track_class : Optional[object] = None
   Giving this will override default Track class, this must inherit Track
clock : Optional[object] = None 
   Clock custom object provides date for track naming and facilitates track id reset every day, preventing overflow and overly large track ids
half : Optional[bool] = True
   Whether to use half precision for deep embedder
bgr : Optional[bool] = True
   Whether frame given to embedder is expected to be BGR or not (RGB)
```



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
