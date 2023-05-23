import os
import logging
from pathlib import Path

import cv2
import numpy as np
import pkg_resources
from mindx.sdk import base
from mindx.sdk.base import Model, Tensor

MOBILENETV2_BOTTLENECK_WTS = pkg_resources.resource_filename(
    "deep_sort_realtime",
    "embedder/weights/mobilenet_v2_tf.om",
)

logger = logging.getLogger(__name__)

INPUT_WIDTH = 224


def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx: min(ndx + bs, l)]


class MobileNetv2_Embedder(object):
    """
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280.

    Params
    ------
    - model_wts_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    - npu_device_id (optional, int) : which device use in npu 

    example
    (change in deep_sort_realtime\deepsort_tracker.py)
    ------

    if embedder == "mobilenet":
        if embedder_npu is True:
            from deep_sort_realtime.embedder.embedder_npu import (
            MobileNetv2_Embedder as Embedder,
            )
            embedder_gpu = False # this may not used
            
            self.embedder = Embedder(
            max_batch_size=4
            bgr=bgr,
            npu_device_id=0,
            model_wts_path=embedder_wts,
            )
        else:
            from deep_sort_realtime.embedder.embedder_pytorch import (
                MobileNetv2_Embedder as Embedder,
            )
            self.embedder = Embedder(
                half=half,
                max_batch_size=16,
                bgr=bgr,
                gpu=embedder_gpu,
                model_wts_path=embedder_wts,
            )

    """

    def __init__(self, model_wts_path=None, max_batch_size=16, bgr=True, npu_device_id=0):

        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        model_wts_path = Path(model_wts_path)
        assert (
            model_wts_path.is_file()
        ), f"Mobilenetv2 model path {model_wts_path} does not exists!"

        self.max_batch_size = max_batch_size
        self.bgr = bgr
        self.device_id = npu_device_id

        base.mx_init()
        self.om_model = Model(str(model_wts_path), self.device_id)

        logger.info("MobileNetV2 Embedder (NPU) for Deep Sort initialised")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")

        zeros = np.zeros((INPUT_WIDTH, INPUT_WIDTH, 3), dtype=np.uint8)

        self.predict([zeros, zeros])  # warmup

    def preprocess(self, np_image):
        """
        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        mindX SDK Tensor

        """
        
        if self.bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image
        np_image_rgb = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
        np_image_rgb = np_image_rgb.astype(np.float32) / 255
        np_image_rgb = np.expand_dims(np_image_rgb, axis=0)

        return np_image_rgb

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1280)

        """
        all_feats = []

        preproc_imgs = [self.preprocess(img) for img in np_images]

        for this_batch in batch(preproc_imgs, bs=self.max_batch_size):
            tensor_list = []
            for tensor in this_batch:
                tensor = Tensor(tensor)
                tensor.to_device(self.device_id)
                tensor_list.append(tensor)

            infer_tensors = base.batch_concat(tensor_list)
            output = self.om_model.infer([infer_tensors])

            for i, output_tensor in enumerate(output):
                output_tensor.to_host()
                output[i] = np.array(output_tensor)
                all_feats.extend(output[i])

        return all_feats
