import unittest
from pathlib import Path

pardir = Path(__file__).parent

import torch
GPU = torch.cuda.is_available()

def test_embedder_generic(Embedder_object, thresh=10, gpu=GPU):
    import cv2
    import numpy as np

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

    small = np.linalg.norm(a - b)
    large = np.linalg.norm(a - c)

    print(f"close: {small} vs diff: {large}")
    assert small < thresh, f"Small: {small} not small enough"
    assert large > thresh, f"Large: {large} not large enough"
    return True


class TestModule(unittest.TestCase):
    def test_embedder_torch(self):
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
        print('Testing pytorch embedder')
        return test_embedder_generic(MobileNetv2_Embedder)

    def test_embedder_torch_cpu(self):
        from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
        print('Testing pytorch embedder')
        return test_embedder_generic(MobileNetv2_Embedder, gpu=False)

    def test_embedder_tf(self):
        from deep_sort_realtime.embedder.embedder_tf import MobileNetv2_Embedder
        print('Testing pytorch embedder in cpu')
        return test_embedder_generic(MobileNetv2_Embedder,)

    def test_embedder_tf_cpu(self):
        from deep_sort_realtime.embedder.embedder_tf import MobileNetv2_Embedder
        print('Testing tf embedder in cpu')
        return test_embedder_generic(MobileNetv2_Embedder, gpu=False)


if __name__ == "__main__":
    unittest.main()
