import time
import logging

import cv2
import numpy as np

from itertools import cycle

try:
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from embedder_pytorch import MobileNetv2_Embedder as Embedder
    from utils.nms import non_max_suppression
except:
    from .deep_sort import nn_matching
    from .deep_sort.detection import Detection
    from .deep_sort.tracker import Tracker
    from .embedder_pytorch import MobileNetv2_Embedder as Embedder
    from .utils.nms import non_max_suppression

log_level = logging.DEBUG
default_logger = logging.getLogger('DeepSORT')
default_logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
default_logger.addHandler(handler)

class DeepSort(object):

    def __init__(self, max_age = 30, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=None, override_track_class=None, clock=None, embedder=True, half=True, bgr=True, logger=None):
        '''
        
        Parameters
        ----------
        max_age : Optional[int] = 30
            Maximum number of missed misses before a track is deleted.
        nms_max_overlap : Optional[float] = 1.0
            Non-maxima suppression threshold: Maximum detection overlap, if is 1.0, nms will be disabled
        max_cosine_distance : Optional[float] = 0.2
            Gating threshold for cosine distance
        nn_budget :  Optional[int] = None
            Maximum size of the appearance descriptors, if None, no budget is enforced
        override_track_class : Optional[object] = None
            Giving this will override default Track class, this must inherit Track
        clock : Optional[object] = None 
            Clock custom object provides date for track naming and facilitates track id reset every day, preventing overflow and overly large track ids. For example clock class, please see `utils/clock.py`
        embedder : Optional[bool] = True
            Whether to use in-built embedder or not. If False, then embeddings must be given during update
        half : Optional[bool] = True
            Whether to use half precision for deep embedder
        bgr : Optional[bool] = True
            Whether frame given to embedder is expected to be BGR or not (RGB)
        logger : Optional[object] = None
            logger object
        '''
        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

        # self.video_info = video_info
        # assert clock is not None
        self.nms_max_overlap = nms_max_overlap
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age, override_track_class=override_track_class, clock=clock)
        if embedder:
            self.embedder = Embedder(half=half, max_batch_size=16, bgr=bgr)
        else:
            self.embedder = None
        self.logger.info('DeepSort Tracker (for OBB) initialised')
        self.logger.info(f'- max age: {max_age}')
        self.logger.info(f'- appearance threshold: {max_cosine_distance}')
        self.logger.info(f'- nms threshold: {"OFF" if self.nms_max_overlap==1.0 else self.nms_max_overlap }')
        self.logger.info(f'- max num of appearance features: {nn_budget}')
        self.logger.info(f'- overriding track class : {"No" if override_track_class is None else "Yes"}' )
        self.logger.info(f'- clock : {"No" if clock is None else "Yes"}' )
        self.logger.info(f'- in-build embedder : {"No" if self.embedder is None else "Yes"}' )

    def update_tracks(self, raw_detections, embeds=None, frame=None):

        """Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        raw_detections : List[ List[float] ]
            List of detections of [x1,y1,x2,y2,x3,y3,x4,y4,confidence]
        embeds : Optional[ List[] ] = None
            List of appearance features corresponding to detections
        frame : Optional [ np.ndarray ] = None
            if embeds not given, Image frame must be given here, in [H,W,C].

        Returns
        -------
        list of track objects (Look into track.py for more info or see "main" section below in this script to see simple example)

        """

        if embeds is None:
            if self.embedder is None:
                raise Exception('Embedder not created during init so embeddings must be given now!')
            if frame is None:
                raise Exception('either embeddings or frame must be given!')

        if embeds is None:
            embeds = self.generate_embeds(frame, raw_detections)
    
        # Proper deep sort detection objects that consist of bbox, confidence and embedding.
        detections = self.create_detections(raw_detections, embeds)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if self.nms_max_overlap < 1.0:
            # nms_tic = time.perf_counter()
            indices = non_max_suppression(
            boxes, self.nms_max_overlap, scores)
            # nms_toc = time.perf_counter()
            # logger.debug(f'nms time: {nms_toc-nms_tic}s')
            detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        return self.tracker.tracks
    
    def generate_embeds(self, frame, raw_dets):
        polygons = []

        detections = np.concatenate(raw_dets)

        for detection in detections:
            points = [int(x) for x in detection[:-1]]
            polygon = [points[x:x+2] for x in range(0, len(points), 2)]
            polygons.append(polygon)

        crops = self.crop_obbs_pad_black(frame, polygons)

        return self.embedder.predict(crops)

    def create_detections(self, raw_dets, embeds):
        embeds_cycle = cycle(embeds)
        detection_list = []

        for j in range(len(raw_dets)):
            try:
                dets = raw_dets[j] # detections for each class
            except:
                import pdb;
                pdb.set_trace()

            for det in dets:
                score = det[-1]

                points = [int(x) for x in det[:8]]
                polygon = [points[x:x+2] for x in range(0, len(points), 2)]
                polygon_mask = np.array([polygon])
                x,y,w,h = cv2.boundingRect(polygon_mask) # in xywh
                if w > 0 and h > 0:
                    x = max(0, x)
                    y = max(0, y)
                    bbox = [x,y,w,h]
                    detection_list.append(Detection(bbox, score, next(embeds_cycle), j))

        return detection_list

    def refresh_track_ids(self):
        self.tracker._next_id

    @staticmethod
    def crop_obbs_pad_black(frame, polygons):
        im_height, im_width = frame.shape[:2]
        masked_obbs = []

        for polygon in polygons:
            mask = np.zeros(frame.shape, dtype=np.uint8)
            polygon_mask = np.array([polygon])
            cv2.fillPoly(mask, polygon_mask, color=(255,255,255))

            # apply the mask
            masked_image = cv2.bitwise_and(frame, mask)

            # crop masked image
            x,y,w,h = cv2.boundingRect(polygon_mask)
            if w > 0 and h > 0:
                crop_l = max(0, x)
                crop_r = min(im_width, x+w)
                crop_t = max(0, y)
                crop_b = min(im_height, y+h)
                cropped = masked_image[crop_t:crop_b, crop_l:crop_r].copy()
                masked_obbs.append(np.array(cropped))

        return masked_obbs

if __name__ == '__main__':
    from utils.clock import Clock
    
    clock = Clock()
    # tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock)
    tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=False)

    impath = '/media/dh/HDD/sample_data/images/cute_doggies.jpg'

    print()
    print('FRAME1')
    frame1 = cv2.imread(impath)
    detections1 = [ ( [0,0,50,50], 0.5, 'person' ), ([50,50, 50, 50], 0.5, 'person') ] 
    embeds1 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.0,1.0,0.5,-0.5])  ]
    # tracks = tracker.update_tracks(detections1, frame=frame1)
    tracks = tracker.update_tracks(detections1, embeds=embeds1)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())

    print()
    print('FRAME2')
    # assume new frame
    frame2 = frame1
    detections2 = [ ( [10,10,60,60], 0.8, 'person' ), ([60,50, 50, 50], 0.7, 'person') ] 
    embeds2 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.1,1.0,0.5,-0.5])  ]
    # tracks = tracker.update_tracks(detections2, frame=frame2)
    tracks = tracker.update_tracks(detections2, embeds=embeds2)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())


    print()
    print('FRAME3')
    # assume new frame
    frame3 = frame1
    detections3 = [ ( [20,20,70,70], 0.8, 'person' ), ([70,50, 50, 50], 0.7, 'person') ] 
    embeds3 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.1,1.0,0.5,-0.5])  ]
    # tracks = tracker.update_tracks(detections3, frame=frame3)
    tracks = tracker.update_tracks(detections3, embeds=embeds3)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())

    print()
    print('FRAME4')
    # assume new frame
    frame4 = frame1
    detections4 = [ ( [10,10,60,60], 0.8, 'person' )] 
    embeds4 = [ np.array([0.1,0.1,0.1,0.1]) ]
    # tracks = tracker.update_tracks(detections4, frame=frame4)
    tracks = tracker.update_tracks(detections4, embeds=embeds4)
    for track in tracks:
        print(track.track_id)
        print(track.to_tlwh())