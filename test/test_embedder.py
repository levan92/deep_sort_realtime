import unittest
from pathlib import Path

pardir = Path(__file__).parent


try:
    import torch

    TORCH_INSTALLED = True
    GPU = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_INSTALLED = False
    CLIP_INSTALLED = False
    GPU = False

if TORCH_INSTALLED:
    try:
        import clip

        CLIP_INSTALLED = True
    except ModuleNotFoundError:
        CLIP_INSTALLED = False

try:
    import tensorflow

    TF_INSTALLED = True
    if not GPU:
        from tensorflow.python.client import device_lib

        gpus = [
            x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
        ]
        GPU = len(gpus) > 0
except ModuleNotFoundError:
    TF_INSTALLED = False


def test_embedder_generic(Embedder_object, thresh=0.2, gpu=GPU):
    import cv2
    import numpy as np

    from deep_sort_realtime.deep_sort.nn_matching import _nn_cosine_distance

    imgpath = pardir / "smallapple.jpg"
    imgpath2 = pardir / "rock.jpg"

    img = cv2.imread(str(imgpath))
    small_angle = 1
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, small_angle, 1.0)
    img_rot = cv2.warpAffine(
        img,
        rot_mat,
        img.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )

    img2 = cv2.imread(str(imgpath2))

    emb = Embedder_object(max_batch_size=4, gpu=gpu)
    a, b, c = emb.predict([img, img_rot, img2])

    small = _nn_cosine_distance([a], [b])[0]
    large = _nn_cosine_distance([a], [c])[0]

    print(f"close: {small} vs diff: {large}")
    assert small < thresh, f"Small: {small} not small enough"
    assert large > thresh, f"Large: {large} not large enough"
    return True


class TestModule(unittest.TestCase):
    @unittest.skipIf(not TORCH_INSTALLED, "Tensorflow is not installed")
    def test_embedder_torch(self):
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

        print("Testing pytorch embedder")
        return test_embedder_generic(MobileNetv2_Embedder)

    @unittest.skipIf(not TORCH_INSTALLED, "Tensorflow is not installed")
    def test_embedder_torch_cpu(self):
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

        print("Testing pytorch embedder")
        return test_embedder_generic(MobileNetv2_Embedder, gpu=False)

    @unittest.skipIf(not TF_INSTALLED, "Tensorflow is not installed")
    def test_embedder_tf(self):
        from deep_sort_realtime.embedder.embedder_tf import MobileNetv2_Embedder

        print("Testing pytorch embedder in cpu")
        return test_embedder_generic(
            MobileNetv2_Embedder,
        )

    @unittest.skipIf(not TF_INSTALLED, "Tensorflow is not installed")
    def test_embedder_tf_cpu(self):
        from deep_sort_realtime.embedder.embedder_tf import MobileNetv2_Embedder

        print("Testing tf embedder in cpu")
        return test_embedder_generic(MobileNetv2_Embedder, gpu=False)

    @unittest.skipIf(not CLIP_INSTALLED, "CLIP is not installed")
    def test_embedder_clip(self):
        from deep_sort_realtime.embedder.embedder_clip import Clip_Embedder

        print("Testing CLIP embedder")
        return test_embedder_generic(Clip_Embedder)

    @unittest.skipIf(not CLIP_INSTALLED, "CLIP is not installed")
    def test_embedder_clip_cpu(self):
        from deep_sort_realtime.embedder.embedder_clip import Clip_Embedder

        print("Testing CLIP embedder")
        return test_embedder_generic(Clip_Embedder, gpu=False)


if __name__ == "__main__":
    unittest.main()
