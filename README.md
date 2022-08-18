# Deep SORT

## Introduction

A more realtime adaptation of Deep SORT.

Adapted from the [official repo of *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT)](https://github.com/nwojke/deep_sort)

See their [paper](https://arxiv.org/abs/1703.07402) for more technical information.

## Dependencies

`requirements.txt` gives the default packages required (installs torch/torchvision to use the default mobilenet embedder), modify accordingly.

Main dependencies are:

- Python3
- NumPy, `pip install numpy`
- SciPy, `pip install scipy`
- cv2, `pip install opencv-python`
- (optional) [Embedder](#appearance-embedding-network) requires Pytorch & Torchvision or Tensorflow 2+
  - `pip install torch torchvision`
  - `pip install tensorflow`
- (optional) To use [CLIP](https://github.com/openai/CLIP) embedder, `pip install git+https://github.com/openai/CLIP.git`

## Install

- from [PyPI](https://pypi.org/project/deep-sort-realtime/) via `pip3 install deep-sort-realtime`
- or, clone this repo & install deep-sort-realtime as a python package using `pip` or as an editable package if you like (`-e` flag)

```bash
cd deep_sort_realtime && pip3 install .
```

- or, download `.whl` file in this repo's [releases](https://github.com/levan92/deep_sort_realtime/releases/latest)

## Run

Example usage:

```python
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=5)
bbs = object_detector.detect(frame) 
tracks = tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
for track in tracks:
    if not track.is_confirmed():
        continue
    track_id = track.track_id
    ltrb = track.to_ltrb()
```

- To add project-specific logic into the `Track` class, you can make a subclass (of `Track`) and pass it in (`override_track_class` argument) when instantiating `DeepSort`.

- Example with your own embedder/ReID model: 

```python
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=5)
bbs = object_detector.detect(frame) # your own object detection
object_chips = chipper(frame, bbs) # your own logic to crop frame based on bbox values
embeds = embedder(object_chips) # your own embedder to take in the cropped object chips, and output feature vectors
tracks = tracker.update_tracks(bbs, embeds=embeds) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class ), also, no need to give frame as your chips has already been embedded
for track in tracks:
    if not track.is_confirmed():
        continue
    track_id = track.track_id
    ltrb = track.to_ltrb()
```

- Look into [`deep_sort_realtime/deepsort_tracker.py`](deep_sort_realtime/deepsort_tracker.py) for more hyperparameters which you can tune to suit your use-case. 

## Getting bounding box of original detection

The original `Track.to_*` methods for retrieving bounding box values returns only the Kalman predicted values. However, in some applications, it is better to return the bb values of the original detections the track was associated to at the current round.

Here we added an `orig` argument to all the `Track.to_*` methods. If `orig` is flagged as `True` and this track is associated to a detection this update round, then the bounding box values returned by the method will be that associated to the original detection. Otherwise, it will still return the Kalman predicted values.

`orig_strict` argument in all the `Track.to_*` methods is only active when `orig` is `True`. Flagging `orig_strict=True` will mean it will output `None` when there's no original detection associated to this track at current frame, otherwise normally it will return Kalman predicted values.

### Storing supplementary info of original detection

Supplementary info can be pass into the track from the detection. `Detection` class now has an `others` argument to store this and pass it to the associate track during update. Can be retrieved through `Track.get_det_supplementary` method. Can be passed in through `others` argument of `DeepSort.update_tracks`, expects to be a list with same length as `raw_detections`. Examples of when you will this includes passing in corresponding instance segmentation masks, to be consumed when iterating through the tracks output. 

## Polygon support

Other than horizontal bounding boxes, detections can now be given as polygons. We do not track polygon points per se, but merely convert the polygon to its bounding rectangle for tracking. That said, if embedding is enabled, the embedder works on the crop around the bounding rectangle, with area not covered by the polygon masked away.

When instantiating a `DeepSort` object (as in `deepsort_tracker.py`), `polygon` argument should be flagged to `True`. See `DeepSort.update_tracks` docstring for details on the polygon format. In polygon mode, the original polygon coordinates are passed to the associated track through the [supplementary info](#storing-supplementary-info-of-original-detection).

## Differences from original repo

- Remove "academic style" offline processing style and implemented it to take in real-time detections and output accordingly.
- Provides both options of using an in-built appearance feature embedder or to provide embeddings during update
- Added pytorch mobilenetv2 as appearance embedder (tensorflow embedder is also available now too).
- Added [CLIP](https://github.com/openai/CLIP) network from OpenAI as embedder (pytorch).
- Skip nms completely in preprocessing detections if `nms_max_overlap == 1.0` (which is the default), in the original repo, nms will still be done even if threshold is set to 1.0 (probably because it was not optimised for speed).
- Now able to override the `Track` class with a custom Track class (that inherits from `Track` class) for custom track logic
- Takes in today's date now, which provides date for track naming and facilities track id reset every day, preventing overflow and overly large track ids when system runs for a long time.
  
  ```python3
  from datetime import datetime
  today = datetime.now().date()
  ```

- Now supports polygon detections. We do not track polygon points per se, but merely convert the polygon to its bounding rectangle for tracking. That said, if embedding is enabled, the embedder works on the crop around the bounding rectangle, with area not covered by the polygon masked away. [Read more here](#polygon-support).
- The original `Track.to_*` methods for retrieving bounding box values returns only the Kalman predicted values. In some applications, it is better to return the bb values of the original detections the track was associated to at the current round. Added a `orig` argument which can be flagged `True` to get that. [Read more here](#getting-bounding-box-of-original-detection).
- Added `get_det_supplementary` method to `Track` class, in order to pass detection related info through the track. [Read more here](#storing-supplementary-info-of-original-detection).
- Other minor adjustments/optimisation of code.

## Highlevel overview of source files in `deep_sort` (from original repo)

In package `deep_sort` is the main tracking code:

- `detection.py`: Detection base class.
- `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
- `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
- `iou_matching.py`: This module contains the IOU matching metric.
- `nn_matching.py`: A module for a nearest neighbor matching metric.
- `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
- `tracker.py`: This is the multi-target tracker class.

## Test

```bash
python3 -m unittest
```

## Appearance Embedding Network

### Pytorch Embedder (default)

Default embedder is a pytorch MobilenetV2 (trained on Imagenet).

For convenience (I know it's not exactly best practice) & since the weights file is quite small, it is pushed in this github repo and will be installed to your Python environment when you install deep_sort_realtime.  

#### CLIP

[CLIP](https://github.com/openai/CLIP) is added as another option of embedder due to its proven flexibility and generalisability. Download the CLIP model weights you want at [deep_sort_realtime/embedder/weights/download_clip_wts.sh](deep_sort_realtime/embedder/weights/download_clip_wts.sh) and store the weights at that directory as well, or you can provide your own CLIP weights through `embedder_wts` argument of the `DeepSort` object.

### Tensorflow Embedder

Available now at `deep_sort_realtime/embedder/embedder_tf.py`, as alternative to (the default) pytorch embedder. Tested on Tensorflow 2.3.1. You need to make your own code change to use it.

The tf MobilenetV2 weights (pretrained on imagenet) are not available in this github repo (unlike the torch one). Download from this [link](https://drive.google.com/file/d/1RBroAFc0tmfxgvrh7iXc2e1EK8TVzXkA/view?usp=sharing) or run [download script](./deep_sort_realtime/embedder/weights/download_tf_wts.sh). You may drop it into `deep_sort_realtime/embedder/weights/` before pip installing.
