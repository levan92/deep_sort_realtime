# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from .application_util import preprocessing
from .application_util import visualization
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker

class DeepSort(object):

    def __init__(self, max_age = 30, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=None):
        '''
        Input Params:
            - nms_max_overlap: Non-maxima suppression threshold: Maximum detection overlap
            - max_cosine_distance: Gating threshold for cosine distance
            - nn_budget: Maximum size of the appearance descriptors, if None, no budget is enforced
        '''
        print('Initialising DeepSort..')
        # self.video_info = video_info
        self.nms_max_overlap = nms_max_overlap
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age = max_age)

    # def update_tracks(sequence_dir, detection_file, output_file, in_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display):
    def update_tracks(self, frame, raw_detections, embeds):

        """Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        sequence_dir : str
            Path to the MOTChallenge sequence directory.
        detection_file : str
            Path to the detections file.
        output_file : str
            Path to the tracking output file. This file will contain the tracking
            results on completion.
        min_confidence : float
            Detection confidence threshold. Disregard all detections that have
            a confidence lower than this value.
        nms_max_overlap: float
            Maximum detection overlap (non-maxima suppression threshold).
        min_detection_height : int
            Detection height threshold. Disregard all detections that have
            a height lower than this value.
        max_cosine_distance : float
            Gating threshold for cosine distance metric (object appearance).
        nn_budget : Optional[int]
            Maximum size of the appearance descriptor gallery. If None, no budget
            is enforced.
        display : bool
            If True, show visualization of intermediate tracking results.

        """
        # frame_info = gather_sequence_info(sequence_dir, detection_file)

        results = []

        # def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = self.create_detections(raw_detections, embeds)
        # detections = create_detections(
            # frame_info["detections"], frame_idx, min_detection_height)
        # detections = [d for d in detections if d['confidence'] >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        return self.tracker.tracks

        # # Update visualization.
        # if display:
        #     image = cv2.imread(
        #         frame_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        #     vis.set_image(image.copy())
        #     vis.draw_detections(detections)
        #     vis.draw_trackers(tracker.tracks)

        # Store results.
        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlwh()
        #     results.append([
        #         frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            # TODO: should we write to file immedi1ately? yes, but not here yo.

        # # Run tracker.
        # if display:
        #     visualizer = visualization.Visualization frame_info, update_ms=5)
        # else:
        #     visualizer = visualization.NoVisualization frame_info)
        # visualizer.run(frame_callback)

        # Store results.
        # TODO, still need to store results

        # f = open(output_file, 'w')
        # for row in results:
        #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
        #         row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


   # def create_detections(detection_mat, frame_idx, min_height=0):
    def create_detections(self, detections, embeds):
        # frame_indices = detection_mat[:, 0].astype(np.int)
        detection_list = []
        for i in range(len(detections)):
            # bbox, confidence, feature = row[2:6], row[6], row[10:]
            detection = detections[i]
            bbox = [detection['rect'][x] for x in 'ltwh']
            confidence = detection['confidence']
            # if bbox[3] < min_height:
                # continue
            detection_list.append(Detection(bbox, confidence, embeds[i]))
        return detection_list

    # def gather_sequence_info(sequence_dir, detection_file):
    #     """Gather sequence information, such as image filenames, detections,
    #     groundtruth (if available).

    #     Parameters
    #     ----------
    #     sequence_dir : str
    #         Path to the MOTChallenge sequence directory.
    #     detection_file : str
    #         Path to the detection file.

    #     Returns
    #     -------
    #     Dict
    #         A dictionary of the following sequence information:

    #         * sequence_name: Name of the sequence
    #         * image_filenames: A dictionary that maps frame indices to image
    #           filenames.
    #         * detections: A numpy array of detections in MOTChallenge format.
    #         * groundtruth: A numpy array of ground truth in MOTChallenge format.
    #         * image_size: Image size (height, width).
    #         * min_frame_idx: Index of the first frame.
    #         * max_frame_idx: Index of the last frame.

    #     """
    #     image_dir = os.path.join(sequence_dir, "img1")
    #     image_filenames = {
    #         int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
    #         for f in os.listdir(image_dir)}
    #     groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    #     # getting Human Objects Detections
    #     # from stored files
    #     detections = None
    #     if detection_file is not None:
    #         detections = np.load(detection_file)
    #     groundtruth = None
    #     if os.path.exists(groundtruth_file):
    #         groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    #     # for real time
    #     # detections = yolo(frame)


    #     if len(image_filenames) > 0:
    #         image = cv2.imread(next(iter(image_filenames.values())),
    #                            cv2.IMREAD_GRAYSCALE)
    #         image_size = image.shape
    #     else:
    #         image_size = None

    #     if len(image_filenames) > 0:
    #         min_frame_idx = min(image_filenames.keys())
    #         max_frame_idx = max(image_filenames.keys())
    #     else:
    #         min_frame_idx = int(detections[:, 0].min())
    #         max_frame_idx = int(detections[:, 0].max())

    #     info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    #     if os.path.exists(info_filename):
    #         with open(info_filename, "r") as f:
    #             line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
    #             info_dict = dict(
    #                 s for s in line_splits if isinstance(s, list) and len(s) == 2)

    #         update_ms = 1000 / int(info_dict["frameRate"])
    #     else:
    #         update_ms = None

    #     feature_dim = detections.shape[1] - 10 if detections is not None else 0
    #     frame_info = {
    #         # "sequence_name": os.path.basename(sequence_dir),
    #         # "image_filenames": image_filenames,
    #         "detection": detection,
    #         "groundtruth": groundtruth,
    #         "image_size": image_size,
    #         # "min_frame_idx": min_frame_idx,
    #         # "max_frame_idx": max_frame_idx,
    #         "feature_dim": feature_dim,
    #         "update_ms": update_ms
    #     }
    #     return frame_info


 

 


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
