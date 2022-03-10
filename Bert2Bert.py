import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
from math import sqrt
import torch as th
from torch.nn.utils import weight_norm
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
import torch
import glob
import re
import copy
import json
import random
import editdistance
# from LCANet_myself import Lipreading
from torch.utils.data import DataLoader
import torch.optim as optim

## Dataset
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad

        with open(file_list, 'r') as f:
            self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)
            self.data.append((vid, items[-4], items[-1]))


    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        vid = np.expand_dims(vid, axis=-1)
        anno = self._load_anno(os.path.join(self.anno_path, 'align', name + '.align'))

        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
## 0312 3012
        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}

    def __len__(self):
        return len(self.data)

    def _load_vid(self, p):
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file.split('_')[1])[0]))
        array = [cv2.imread(os.path.join(p, file), cv2.IMREAD_GRAYSCALE) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])
        return ''.join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer

## ===============


def dataset2dataloader(dataset, num_workers=6, shuffle=True):
    return DataLoader(dataset,
        batch_size = 8,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)


def train(model):
    video_path = '/home/max/Desktop/video'
    train_list = f'data_new/unseen_train.txt'
    val_list = f'data_new/unseen_val.txt'
    anno_path = '/home/max/Desktop/grid_corpus/align'

    dataset = MyDataset(video_path, anno_path, train_list, 75, 200)
    train_loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(), lr = 3e-4)

    for (i_iter, input) in enumerate(train_loader):
        model.train()
        vid = input.get('vid').cuda()
        txt = input.get('txt').cuda()
        vid_len = input.get('vid_len').cuda()
        txt_len = input.get('txt_len').cuda()
        optimizer.zero_grad()
        y = model(vid, txt)
        print(y)
        break
class BERT2GPT(nn.Module):
    def __init__(self):
        super(BERT2GPT, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def forward(self, x, txt):
        x = self.frontend3D(x)
        x = x.flatten()
        return self.model(input_idx=self.tokenizer(x, return_tensors="pt"), labels=self.tokenizer(txt, return_tensors="pt"))


if(__name__ == '__main__'):
    
    # input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
    # labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
    # outputs = model(input_ids=input_ids, labels=input_ids)
    # loss, logits = outputs.loss, outputs.logits
    # model = Lipreading().cuda()
    model = BERT2GPT().cuda()
    train(model)
