import torch
from torchvision.transforms import transforms
import cv2
import math
# import time
import numpy as np
import os 
# if __name__ == '__main__':
from mobilenetv2.mobilenetv2_bottle import MobileNetV2_bottle
# else:
    # from .mobilenetv2.mobilenetv2_bottle import MobileNetV2_bottle
DIR = os.path.dirname(os.path.realpath(__file__))

# MOBILENETV2_BOTTLENECK_TORCH_MODEL =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottle_py35.pt")
MOBILENETV2_BOTTLENECK_WTS =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottleneck_wts.pt")
INPUT_WIDTH = 224

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
    def __init__(self, model_wts_path = None, half=True):
        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        assert os.path.exists(model_wts_path),'Mobilenetv2 model path does not exists!'
        self.model = MobileNetV2_bottle(input_size=INPUT_WIDTH, width_mult=1.)
        self.model.load_state_dict(torch.load(model_wts_path))
        self.model.cuda() #loads model to gpu
        self.model.eval() #inference mode, deactivates dropout layers
        self.half = half
        if self.half:
            self.model.half()
        print('MobileNetV2 Embedder for Deep Sort initialised!')
        # tic = time.time()
        zeros = np.zeros((100, 100, 3))
        self.predict([zeros]) #warmup
        # toc = time.time()
        # print('warm up time: {}s'.format(toc - tic))

    def predict(self, np_image_bgr_batch, batch_size = 16):
        '''
        batch inference

        Params
        ------
        np_image_bgr_batch : list of ndarray
            list of (H x W x C) in BGR
        
        (optional) batch_size : int

        Returns
        ------
        list of features (np.array with dim = 1280)

        '''
        total_size = len(np_image_bgr_batch)
        batch_size = 16

        split_batches = []
        for i in range(0, total_size, batch_size):
            split_batches.append(np_image_bgr_batch[i:i+batch_size])
        all_feats = []
        remainder = total_size
        # For each batch
        for i in range(math.ceil(total_size / batch_size)):
            input_batch = torch.zeros((min(batch_size, remainder), 3, INPUT_WIDTH, INPUT_WIDTH))
            # For each img in batch
            for k, img in enumerate(split_batches[i]):
                img = preprocess(img)
                input_batch[k] = img
            # Batch inference
            # tic = time.time()
            input_batch = input_batch.cuda()
            # toc = time.time()
            # print('input to cuda time:{}s'.format(toc - tic))
            # tic = time.time()
            if self.half:
                input_batch = input_batch.half()
            output = self.model.forward(input_batch)
            # toc = time.time()
            # print('real inference time: {}s'.format(toc - tic))
            all_feats.extend(output.cpu().data.numpy())
            remainder = total_size - batch_size
        return all_feats

if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    # impath = '/home/levan/Pictures/auba.jpg'
    impath = '/home/dh/Pictures/dog_two.jpg'
    # impath = '/home/levan/Pictures/gudeicebear.png'
    auba = cv2.imread(impath)
    
    tic = time.time()
    emb = MobileNetv2_Embedder()
    toc = time.time()
    print('loading time: {}s'.format(toc - tic))

    aubas = [auba] * 1
    tic = time.time()
    feats = emb.predict(aubas)
    toc = time.time()
    print('whole inference 1 time: {}s'.format(toc - tic))
    print(np.shape(feats))

    # aubas = [auba] * 8
    # tic = time.time()
    # feats = emb.predict(aubas)
    # toc = time.time()
    # print('inference 8 time: {}s'.format(toc - tic))
    # print(np.shape(feats))
    
    # aubas = [auba] * 16
    # tic = time.time()
    # feats = emb.predict(aubas)
    # toc = time.time()
    # print('inference 16 time: {}s'.format(toc - tic))
    # print(np.shape(feats))

    # aubas = [auba] * 32
    # tic = time.time()
    # feats = emb.predict(aubas)
    # toc = time.time()
    # print('inference 32 time: {}s'.format(toc - tic))
    # print(np.shape(feats))