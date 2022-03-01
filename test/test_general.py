import os
import unittest
from datetime import datetime

try:
    import torch

    GPU = torch.cuda.is_available() and not os.environ.get("USE_CPU")
    TORCH_INSTALLED = True
except ModuleNotFoundError:
    GPU = False
    TORCH_INSTALLED = False

try:
    import clip

    CLIP_INSTALLED = True
except ModuleNotFoundError:
    CLIP_INSTALLED = False


class TestModule(unittest.TestCase):
    @unittest.skipIf(not TORCH_INSTALLED, "Tensorflow is not installed")
    def test_base(self):

        from deep_sort_realtime.deepsort_tracker import DeepSort

        import numpy as np

        today = datetime.now().date()
        tracker = DeepSort(
            max_age=30,
            nn_budget=100,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            # embedder_wts='deep_sort_realtime/embedder/weights/mobilenetv2_bottleneck_wts.pt',
            embedder_gpu=GPU,
            today=today,
        )

        print()
        print("FRAME1")
        frame1 = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        detections1 = [
            ([0, 0, 50, 50], 0.5, "person"),
            ([50, 50, 50, 50], 0.5, "person"),
        ]

        tracks = tracker.update_tracks(
            detections1, frame=frame1, today=datetime.now().date()
        )
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        return True

    @unittest.skipIf(not CLIP_INSTALLED, "CLIP is not installed")
    def test_clip(self):

        from deep_sort_realtime.deepsort_tracker import DeepSort

        import numpy as np

        today = datetime.now().date()
        tracker = DeepSort(
            max_age=30,
            nn_budget=100,
            nms_max_overlap=1.0,
            embedder="clip_ViT-B/32",
            # embedder_wts='deep_sort_realtime/embedder/weights/ViT-B-32.pt',
            embedder_gpu=GPU,
            today=today,
        )

        print()
        print("FRAME1")
        frame1 = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        detections1 = [
            ([0, 0, 50, 50], 0.5, "person"),
            ([50, 50, 50, 50], 0.5, "person"),
        ]

        tracks = tracker.update_tracks(
            detections1, frame=frame1, today=datetime.now().date()
        )
        for track in tracks:
            print(track.track_id)
            print(track.to_tlwh())

        return True


if __name__ == "__main__":
    unittest.main()
