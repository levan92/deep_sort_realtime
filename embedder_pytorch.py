import os 
import logging

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from mobilenetv2.mobilenetv2_bottle import MobileNetV2_bottle

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

class MobileNetv2_Embedder(object):
    '''
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280. 

    Params
    ------
    - model_wts_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - half (optional, Bool) : boolean flag to use half precision or not, defaults to True
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True

    '''
    def __init__(self, model_wts_path = None, half=True, max_batch_size = 16, bgr=True):
        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        assert os.path.exists(model_wts_path),'Mobilenetv2 model path does not exists!'
        self.model = MobileNetV2_bottle(input_size=INPUT_WIDTH, width_mult=1.)
        self.model.load_state_dict(torch.load(model_wts_path))
        self.model.cuda() #loads model to gpu
        self.model.eval() #inference mode, deactivates dropout layers

        self.max_batch_size = max_batch_size
        self.bgr = bgr

        self.half = half
        if self.half:
            self.model.half()
        logger.info('MobileNetV2 Embedder for Deep Sort initialised')
        logger.info(f'- half precision: {self.half}')
        logger.info(f'- max batch size: {self.max_batch_size}')
        logger.info(f'- expects BGR: {self.bgr}')

        # tic = time.time()
        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros]) #warmup
        # toc = time.time()
        # print('warm up time: {}s'.format(toc - tic))

    # preproc_transforms = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize(INPUT_WIDTH),
    #         # transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    def preprocess(self, np_image):
        '''
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        '''
        if self.bgr:
            np_image_rgb = np_image[...,::-1]
        else:
            np_image_rgb = np_image

        # tic = time.perf_counter()
        input_image = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = trans(input_image)
        input_image = input_image.view(1,3,INPUT_WIDTH,INPUT_WIDTH)
        
        # tic2 = time.perf_counter()
        # input_tensor = self.preproc_transforms(np_image_rgb)
        # input_tensor = input_tensor.unsqueeze(0)
        # toc = time.perf_counter()
        # logger.debug(f'cv2resize:{input_image.size()}')
        # logger.debug(f'cv2resize:{tic2-tic}')
        # logger.debug(f'torch trans:{input_tensor.size()}')
        # logger.debug(f'torch trans:{toc-tic2}')

        # def inverse_normalize(tensor, mean, std):
        #     for t, m, s in zip(tensor, mean, std):
        #         t.mul_(s).add_(m)
        #     return tensor
        # input1 = inverse_normalize(tensor=input_image[0], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # input2 = inverse_normalize(tensor=input_tensor[0], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # show1 = input1.data.numpy().transpose(1,2,0)*255
        # show1 = show1.astype(dtype=np.uint8)
        # show2 = input2.data.numpy().transpose(1,2,0)*255
        # show2 = show2.astype(dtype=np.uint8)
        # print(show1[95,95])
        # print(show2[95,95])
        # cv2.imshow('cv2resize', show1)
        # cv2.imshow('torchtrans', show2)
        # cv2.waitKey(0)
        # assert torch.all(input_image.eq(input_tensor))
        # import pdb; pdb.set_trace()
        
        return input_image

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
    # emb = MobileNetv2_Embedder(half=False, max_batch_size=32)
    emb = MobileNetv2_Embedder(half=True, max_batch_size=32)
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
        print(f'inference BS{bs} avrg time: {dur/reps:0.4f}s')
