import pickle
import random
import cv2
import torch
import os

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from options import pickle_file, IGNORE_ID
#from utils import extract_feature

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img):
    for i in range(batch_img.shape[0]):
        if(random.random() < 0.05 and 0 < i):
            batch_img[i] = batch_img[i - 1]
    return batch_img
    
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


def build_LFR_features(inputs, m, n):
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


class AiShellDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        #data = dict()
        self.split = split
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data[split]
        length = 0
        for sample in self.samples:
            trn = sample['trn']
            #self.trns.append({'trn':trn})
            length_sample = len(trn)
            if length_sample >= length:
               length = length_sample
               
        #data['train'] = self.trns
        #with open(pickle_file, 'wb') as file:
            #pickle.dump(data, file)
               
        print('the max_length is: ', length)
        self.samples = list(filter(lambda data: os.path.exists(data['images']) and 0<len(os.listdir(data['images'])) <= 75, self.samples))
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['wave']
        images = sample['images']
        trn = sample['trn']
        #print(len(trn))

        ###images
        if len(os.listdir(images)) == 0:
            print(images, len(os.listdir(images)))
        vid = self._load_vid(images)
        vid = self._padding(vid, 75)

        if self.split == 'train':
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        vid = ColorNormalize(vid)
        
        #vid = np.expand_dims(vid, axis=3)
        trn = np.pad(trn, (0, 31-len(trn)), 'constant', constant_values=-1)
        #return feature, trn
        #print(vid.shape)
        return torch.FloatTensor(vid), torch.LongTensor(trn)

    def __len__(self):
        return len(self.samples)

    def _load_vid(self, p):
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file.split('_')[1])[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (120, 120), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array



    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)