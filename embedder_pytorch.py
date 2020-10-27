import os 
import logging

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

try:
    from mobilenetv2.mobilenetv2_bottle import MobileNetV2_bottle
except:
    from .mobilenetv2.mobilenetv2_bottle import MobileNetV2_bottle

log_level = logging.DEBUG
logger = logging.getLogger('Embedder for Deepsort')
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

DIR = os.path.dirname(os.path.realpath(__file__))
# MOBILENETV2_BOTTLENECK_TORCH_MODEL =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottle_py35.pt")
MOBILENETV2_BOTTLENECK_WTS =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottleneck_wts.pt")
INPUT_WIDTH = 224

def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx:min(ndx + bs, l)]

def preprocess(np_image_bgr):
    '''
    Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

    Parameters
    ----------
    np_image_bgr : ndarray
        (H x W x C) in BGR

    Returns
    -------
    Torch Tensor

    '''
    # tic = time.time()
    np_image_rgb = np_image_bgr[...,::-1]
    # toc = time.time()
    # print('flip channel time: {}s'.format(toc - tic))
    # tic = time.time()
    np_image_rgb = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
    # toc = time.time()
    # print('resize time: {}s'.format(toc - tic))
    # preproc = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # tic = time.time()
    input_image = transforms.ToTensor()(np_image_rgb)
    # input_image = preproc(np_image_rgb)
    # toc = time.time()
    # print('toTorchTensor & Norm time: {}s'.format(toc - tic))
    # tic = time.time()
    input_image = input_image.view(1,3,INPUT_WIDTH,INPUT_WIDTH)
    # toc = time.time()
    # print('reshape time: {}s'.format(toc - tic))

    return input_image

class MobileNetv2_Embedder(object):
    '''
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280. 
    '''
    def __init__(self, model_wts_path = None, half=True, max_batch_size = 16):
        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        assert os.path.exists(model_wts_path),'Mobilenetv2 model path does not exists!'
        self.model = MobileNetV2_bottle(input_size=INPUT_WIDTH, width_mult=1.)
        self.model.load_state_dict(torch.load(model_wts_path))
        self.model.cuda() #loads model to gpu
        self.model.eval() #inference mode, deactivates dropout layers

        self.max_batch_size = max_batch_size
        self.half = half
        if self.half:
            self.model.half()
        logger.info('MobileNetV2 Embedder for Deep Sort initialised')
        logger.info(f'- half precision: {self.half}')
        logger.info(f'- max batch size: {self.max_batch_size}')
        # tic = time.time()
        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros]) #warmup
        # toc = time.time()
        # print('warm up time: {}s'.format(toc - tic))


    def predict(self, np_image_bgr_batch):
        '''
        batch inference

        Params
        ------
        np_image_bgr_batch : list of ndarray
            list of (H x W x C) in BGR
        
        Returns
        ------
        list of features (np.array with dim = 1280)

        '''
        all_feats = []

        preproc_imgs = [ preprocess(img) for img in np_image_bgr_batch ]

        for this_batch in batch(preproc_imgs, bs=self.max_batch_size):
            this_batch = torch.cat(this_batch, dim=0)
            this_batch = this_batch.cuda()
            if self.half:
                this_batch = this_batch.half()
            output = self.model.forward(this_batch)
            
            all_feats.extend(output.cpu().data.numpy())

        return all_feats

if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    impath = '/media/dh/HDD/sample_data/images/cute_doggies.jpg'
    auba = cv2.imread(impath)
    
    tic = time.time()
    emb = MobileNetv2_Embedder(half=False, max_batch_size=32)
    # emb = MobileNetv2_Embedder(half=True, max_batch_size=32)
    toc = time.time()
    print(f'loading time: {toc - tic:0.4f}s')

    bses = [1,16,32,100]
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
        print(f'avrg inference {bs} time: {dur/reps:0.4f}s')
