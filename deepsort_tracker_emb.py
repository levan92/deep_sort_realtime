import cv2
import numpy as np

if __name__ == '__main__':
    from application_util import preprocessing
    from application_util import visualization
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from embedder_pytorch import MobileNetv2_Embedder as Embedder
else:
    from .application_util import preprocessing
    from .application_util import visualization
    from .deep_sort import nn_matching
    from .deep_sort.detection import Detection
    from .deep_sort.tracker import Tracker
    from .embedder_pytorch import MobileNetv2_Embedder as Embedder

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
        self.embedder = Embedder()
        print('DeepSort Tracker initialised!')

    def update_tracks(self, frame, raw_detections):

        """Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        frame : ndarray
            Path to the MOTChallenge sequence directory.
        raw_detections : list
            List of triples ( [left,top,w,h] , confidence, detection_class)

        Returns
        -------
        list of track objects (Look into track.py for more info or see "main" section below in this script to see simple example)

        """

        results = []

        embeds = self.generate_embeds(frame, raw_detections)
        # Proper deep sort detection objects that consist of bbox, confidence and embedding.
        detections = self.create_detections(frame, raw_detections, embeds)

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

    def generate_embeds(self, frame, raw_dets):
        crops = []
        for detection in raw_dets:
            if detection is None:
                continue
            l,t,w,h = [int(x) for x in detection[0]]
            crops.append(frame[ t:t+h-1, l:l+w-1 ])
        return self.embedder.predict(crops)

    def create_detections(self, frame, raw_dets, embeds):
        detection_list = []
        for i in range(len(embeds)):
            detection_list.append(Detection(raw_dets[i][0], raw_dets[i][1], embeds[i]))
        return detection_list

if __name__ == '__main__':
    tracker = DeepSort(max_age = 30, nn_budget=100)

    impath = '/home/levan/Pictures/auba.jpg'
    
    print()
    print('FRAME1')
    frame1 = cv2.imread(impath)
    detections1 = [ ( [0,0,50,50], 0.5 ), ([50,50, 50, 50], 0.5) ] 
    tracks = tracker.update_tracks(frame1, detections1)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())

    print()
    print('FRAME2')
    # assume new frame
    frame2 = frame1
    detections2 = [ ( [10,10,60,60], 0.8 ), ([60,50, 50, 50], 0.7) ] 
    tracks = tracker.update_tracks(frame2, detections2)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())