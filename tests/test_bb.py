import unittest

MAGIC_TEST_IMAGE = '/media/dh/HDD/sample_data/images/cute_doggies.jpg'

class TestModule(unittest.TestCase):
    def test_hbb(self):

        from deepsort_tracker import DeepSort
        from utils.clock import Clock

        import cv2
        import numpy as np

        clock = Clock()
        tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=False)

        print()
        print('FRAME1')
        frame1 = cv2.imread(MAGIC_TEST_IMAGE)
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
        
        return True

    def test_obb(self):

        from deepsort_tracker import DeepSort
        from utils.clock import Clock

        import cv2
        import numpy as np

        clock = Clock()
        tracker = DeepSort(max_age = 30, nn_budget=100, nms_max_overlap=1.0, clock=clock, embedder=True, polygon=True)

        print()
        print('FRAME1')
        frame1 = cv2.imread(MAGIC_TEST_IMAGE)
        detections1 = [ [ np.array([0,0,10,0,10,10,0,10,0.5]), np.array([20,20,30,20,30,30,20,30,0.5]) ], np.zeros((0, 9)), [] ]
        tracks = tracker.update_tracks(detections1, frame=frame1)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME2')
        # assume new frame
        frame2 = frame1
        detections2 = [ [ np.array([10,0,20,0,30,10,10,10,0.5]), np.array([20,30,30,30,30,40,20,40,0.5]) ], np.zeros((0, 9)), [] ]
        tracks = tracker.update_tracks(detections2, frame=frame2)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME3')
        # assume new frame
        frame3 = frame1
        detections3 = [ [np.array([0,0,10,0,10,10,5,15,0,10,0.5])], [], [np.array([20,20,30,20,30,30,20,30,0.5])] ]
        tracks = tracker.update_tracks(detections3, frame=frame3)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        print()
        print('FRAME4')
        # assume new frame
        frame4 = frame1
        detections4 = [ [ np.array([0,0,10,0,10,10,0,10,0.5]), np.array([20,20,30,20,30,30,20,30,0.5]) ], [], [] ]
        tracks = tracker.update_tracks(detections4, frame=frame4)
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        return True

if __name__ == '__main__':
    unittest.main()