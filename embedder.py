import torch
from torchvision.transforms import transforms
import cv2
# import time
import os 
DIR = os.path.dirname(os.path.realpath(__file__))

MOBILENETV2_BOTTLENECK_TORCH_MODEL =os.path.join(DIR,"mobilenetv2/mobilenetv2_bottle_py35.pt")
INPUT_WIDTH = 112

def preprocess(np_image_bgr):
    '''
    Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

    Parameters
    ----------
    np_image : ndarray
        (H x W x C) in BGR

    Returns
    -------
    Torch Tensor

    '''
    np_image_rgb = np_image_bgr[...,::-1]
    np_image_rgb = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
    preproc = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # tic = time.time()
    input_image = preproc(np_image_rgb)
    # toc = time.time()
    # print('preproc time: {}s'.format(toc - tic))

    input_image = input_image.view(1,3,INPUT_WIDTH,INPUT_WIDTH)
    # input_image = input_image.cuda()
    return input_image

class MobileNetv2_Embedder(object):
    '''
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280. 
    '''
    def __init__(self, model_path = None):
        if model_path is None:
            model_path = MOBILENETV2_BOTTLENECK_TORCH_MODEL
        assert os.path.exists(model_path),'Mobilenetv2 model path does not exists!'
        self.model = torch.load(model_path)
        self.model.cuda() #loads model to gpu
        self.model.eval() #inference mode, deactivates dropout layers 
        print('MobileNetV2 Embedder for Deep Sort initialised!')

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
        for i in range(total_size // batch_size + 1):
            input_batch = torch.zeros((min(batch_size, remainder), 3, INPUT_WIDTH, INPUT_WIDTH))
            # For each img in batch
            for k, img in enumerate(split_batches[i]):
                img = preprocess(img)
                input_batch[k] = img
            # Batch inference
            input_batch = input_batch.cuda()
            # tic = time.time()
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
    impath = '/home/levan/Pictures/auba.jpg'
    auba = cv2.imread(impath)
    aubas = [auba] * 1
    
    tic = time.time()
    emb = MobileNetv2_Embedder()
    toc = time.time()
    print('loading time: {}s'.format(toc - tic))

    tic = time.time()
    feats = emb.predict(aubas)
    toc = time.time()
    print('inference time: {}s'.format(toc - tic))
    
    print(feats)
    print(np.shape(feats))