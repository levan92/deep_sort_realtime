import unittest
import time

class TestModule(unittest.TestCase):
    def test_hbb(self):

        from deepsort_tracker import DeepSort
        from utils.clock import Clock

        import cv2
        import numpy as np

        clock = Clock()
        # tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=False)
        tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=True)

        tic = time.perf_counter()
        print()
        print('FRAME1')
        frame1 = np.ones((1080,1920,3), dtype=np.uint8) * 255
        detections1 = [ ( [0,0,50,50], 0.5, 'person' ), ([50,50, 50, 50], 0.5, 'person') ] 
        embeds1 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.0,1.0,0.5,-0.5])  ]
        # tracks = tracker.update_tracks(detections1, embeds=embeds1)
        tracks = tracker.update_tracks(detections1, frame=frame1)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME2')
        # assume new frame
        frame2 = frame1
        detections2 = [ ( [10,10,60,60], 0.8, 'person' ), ([60,50, 50, 50], 0.7, 'person') ] 
        embeds2 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.1,1.0,0.5,-0.5])  ]
        # tracks = tracker.update_tracks(detections2, embeds=embeds2)
        tracks = tracker.update_tracks(detections2, frame=frame2)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME3')
        # assume new frame
        frame3 = frame1
        detections3 = [ ( [20,20,70,70], 0.8, 'person' ), ([70,50, 50, 50], 0.7, 'person') ] 
        embeds3 = [ np.array([0.1,0.1,0.1,0.1]), np.array([-1.1,1.0,0.5,-0.5])  ]
        # tracks = tracker.update_tracks(detections3, embeds=embeds3)
        tracks = tracker.update_tracks(detections3, frame=frame3)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME4')
        # assume new frame
        frame4 = frame1
        detections4 = [ ( [10,10,60,60], 0.8, 'person' )] 
        embeds4 = [ np.array([0.1,0.1,0.1,0.1]) ]
        # tracks = tracker.update_tracks(detections4, embeds=embeds4)
        tracks = tracker.update_tracks(detections4, frame=frame4)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())
        
        toc = time.perf_counter()
        print(f'Avrg Duration per update: {(toc-tic)/4}')
        return True

    def test_obb(self):

        from deepsort_tracker import DeepSort
        from utils.clock import Clock

        import cv2
        import numpy as np

        clock = Clock()
        tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=True, polygon=True)

        tic = time.perf_counter()

        print()
        print('FRAME1')
        frame1 = np.ones((1080,1920,3), dtype=np.uint8) * 255

        detections1 = [
            [[0,0,10,0,10,10,0,10],[20,20,30,20,30,30,20,30]],
            [0,1],
            [0.5,0.5]
        ]
        tracks = tracker.update_tracks(detections1, frame=frame1)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME2')
        # assume new frame
        frame2 = frame1
        detections2 = [
            [[0,0,10,0,15,10,0,15],[25,20,30,20,30,30,25,30]],
            [0,1],
            [0.5,0.6]
        ]
        tracks = tracker.update_tracks(detections2, frame=frame2)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME3')
        # assume new frame
        frame3 = frame1
        detections3 = [
            [[0,0,10,0,15,10,10,15],[20,20,30,20,30,30,25,30]],
            [0,3],
            [0.5,0.6]
        ]
        tracks = tracker.update_tracks(detections3, frame=frame3)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME4')
        # assume new frame
        frame4 = frame1
        detections4 = [
            [[0,5,15,5,15,10,10,25],[20,20,30,20,30,30,25,30]],
            [3,3],
            [0.9,0.6]
        ]
        tracks = tracker.update_tracks(detections4, frame=frame4)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())
        
        toc = time.perf_counter()
        print(f'Avrg Duration per update: {(toc-tic)/4}')

        return True

if __name__ == '__main__':
    unittest.main()