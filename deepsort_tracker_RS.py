# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from .application_util import preprocessing
from .application_util import visualization
from .deep_sort import nn_matching
from .deep_sort.detection_RS import Detection_RS
from .deep_sort.tracker_RS import Tracker_RS

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
        self.tracker = Tracker_RS(metric, max_age = max_age)

    def update_tracks(self, frame, raw_detections, embeds):

        results = []

        # Load image and generate detections.
        detections = self.create_detections(raw_detections, embeds)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        return self.tracker.tracks, self.tracker.del_tracks_ids

    def create_detections(self, detections, embeds):
        detection_list = []
        for i in range(len(detections)):
            cl, confidence, ltrb = detections[i]

            # in LTWH
            bbox = [ ltrb[0], ltrb[1], 
                    (ltrb[2] - ltrb[0] + 1), 
                    (ltrb[3] - ltrb[1] + 1) ]
            detection_list.append(Detection_RS(cl.decode("utf-8"), bbox, confidence, embeds[i]))
        return detection_list
