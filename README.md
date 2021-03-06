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
- Include this repo as submodule
- `deepsort_tracker.py` is your main point of entry

```python
from deep_sort.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=30, nn_budget=70, override_track_class=None)
bbs = object_detector.detect(frame)
tracks = trackers.update_tracks(bbs, frame=frame)
for track in tracks:
   track_id = track.track_id
   ltrb = track.to_ltrb()
```

- To add project-specific logic into the `Track` class, you can make a subclass (of `Track`) and pass it in (`override_track_class` argument) when instantiating `DeepSort`.

## Getting bounding box of original detection

The original `Track.to_*` methods for retrieving bounding box values returns only the Kalman predicted values. However, in some applications, it is better to return the bb values of the original detections the track was associated to at the current round. 

Here we added an `orig` argument to all the `Track.to_*` methods. If `orig` is flagged as `True` and this track is associated to a detection this update round, then the bounding box values returned by the method will be that associated to the original detection. Otherwise, it will still return the Kalman predicted values.

### Storing supplementary info of original detection 

Supplementary info can be pass into the track from the detection. `Detection` class now has an `others` argument to store this and pass it to the associate track during update. Can be retrieved through `Track.get_det_supplementary` method.


## Polygon support

Other than horizontal bounding boxes, detections can now be given as polygons. We do not track polygon points per se, but merely convert the polygon to its bounding rectangle for tracking. That said, if embedding is enabled, the embedder works on the crop around the bounding rectangle, with area not covered by the polygon masked away. 

When instantiating a `DeepSort` object (as in `deepsort_tracker.py`), `polygon` argument should be flagged to `True`. See `DeepSort.update_tracks` docstring for details on the polygon format. In polygon mode, the original polygon coordinates are passed to the associated track through the [supplementary info](#storing-supplementary-info-of-original-detection). 


## Differences from original repo

- Remove "academic style" offline processing style and implemented it to take in real-time detections and output accordingly.
- Provides both options of using an in-built appearance feature embedder or to provide embeddings during update
- Added (pytorch) mobilenetv2 as embedder (torch ftw).
- Skip nms completely in preprocessing detections if `nms_max_overlap == 1.0` (which is the default), in the original repo, nms will still be done even if threshold is set to 1.0 (probably because it was not optimised for speed).
- Now able to override the `Track` class with a custom Track class (that inherits from `Track` class) for custom track logic 
- Now takes in a "clock" object (see `utils/clock.py` for example), which provides date for track naming and facilities track id reset every day, preventing overflow and overly large track ids when system runs for a long time.
- Now supports polygon detections. We do not track polygon points per se, but merely convert the polygon to its bounding rectangle for tracking. That said, if embedding is enabled, the embedder works on the crop around the bounding rectangle, with area not covered by the polygon masked away. [Read more here](#polygon-support).
- The original `Track.to_*` methods for retrieving bounding box values returns only the Kalman predicted values. In some applications, it is better to return the bb values of the original detections the track was associated to at the current round. Added a `orig` argument which can be flagged `True` to get that. [Read more here](#getting-bounding-box-of-original-detection).
- Added `get_det_supplementary` method to `Track` class, in order to pass detection related info through the track. [Read more here](#storing-supplementary-info-of-original-detection).
- Other minor adjustments.

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

## Test

```bash
python3 -m tests.test_bb
```

## Misc
Ignore `deepsort_tracker_emb_dict.py` please, that is WIP. 
