import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance

from options import device, vocab_size, sos_id, eos_id, print_freq
#from data_gen import AiShellDataset, pad_collate
from data_gen import AiShellDataset
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.video_frontend import visual_frontend
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
if __name__ == '__main__':
    global args
    args = parse_args()

    visual_model = visual_frontend(hiddenDim=512, embedSize=256)
        
    encoder = Encoder(512, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)

    model = Transformer(encoder, decoder, visual_model)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('total parameters: ', pytorch_total_params)
    print('trainable total parameters: ', pytorch_total_params_trainable)


