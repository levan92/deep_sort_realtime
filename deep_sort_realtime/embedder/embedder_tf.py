import os 
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

DIR = os.path.dirname(os.path.realpath(__file__))
# MOBILENETV2_BOTTLENECK_TORCH_MODEL =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottle_py35.pt")
MOBILENETV2_BOTTLENECK_WTS =os.path.join(DIR,"mobilenetv2_tf/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5")

log_level = logging.DEBUG
logger = logging.getLogger('Embedder for Deepsort')
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

INPUT_WIDTH = 224

def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx:min(ndx + bs, l)]

def get_mobilenetv2_with_preproc(wts='imagenet'):
    i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    full_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=None, 
        weights=str(wts),
        classifier_activation=None, 
    )
    core_model = tf.keras.Model(full_model.input, full_model.layers[-2].output)

    x = core_model(x)

    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    return model

class MobileNetv2_Embedder(object):
    '''
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280. 

    Params
    ------
    - model_wts_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True

    '''
    def __init__(self, model_wts_path = None, max_batch_size = 16, bgr=True):
        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        model_wts_path = Path(model_wts_path)
        assert model_wts_path.is_file(),f'Mobilenetv2 model path {model_wts_path} does not exists!'

        self.model = get_mobilenetv2_with_preproc(wts=model_wts_path)

        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info('MobileNetV2 Embedder (tf) for Deep Sort initialised')
        logger.info(f'- max batch size: {self.max_batch_size}')
        logger.info(f'- expects BGR: {self.bgr}')

        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros, zeros]) #warmup

    def preprocess(self, np_image):
        '''
        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        TF Tensor

        '''
        if self.bgr:
            np_image_rgb = np_image[...,::-1]
        else:
            np_image_rgb = np_image
        np_image_rgb = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
        return tf.convert_to_tensor(np_image_rgb)

    def predict(self, np_images):
        '''
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr
        
        Returns
        ------
        list of features (np.array with dim = 1280)

        '''
        all_feats = []

        preproc_imgs = [ self.preprocess(img) for img in np_images ]

        for this_batch in batch(preproc_imgs, bs=self.max_batch_size):
            this_batch = tf.stack(this_batch, axis=0)
            output = self.model(this_batch)
            all_feats.extend(output.numpy())

        return all_feats


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    impath = '/media/dh/HDD/sample_data/images/cute_doggies.jpg'
    auba = cv2.imread(impath)
    
    tic = time.time()
    emb = MobileNetv2_Embedder(max_batch_size=16)
    toc = time.time()
    print(f'loading time: {toc - tic:0.4f}s')

    bses = [1,16]
    reps = 100
    for bs in bses:
        aubas = [auba] * bs
        dur = 0
        for _ in range(reps):
            tic = time.time()
            feats = emb.predict(aubas)
            toc = time.time()
            dur += toc - tic
        print(np.shape(feats))
        print(f'inference BS{bs} avrg time: {dur/reps:0.4f}s')
