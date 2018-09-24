from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
import dlib
from models.basenet import SqueezeNet
import matplotlib.pyplot as plt 
parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('-img', '--image', default='face76', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint/1011/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
mean = np.asarray([100, 100, 100])
std = np.asarray([50.0, 50.0, 50.0])



def load_model():
    model = SqueezeNet(136).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    out_size = 224
    model = load_model()
    model = model.eval()
    img = cv2.imread('dataset/dd/test/Jan_Peter_Balkenende_11.jpg')
    img = cv2.resize(img,(224,224))
    raw_img = img
    img = img/255.0
    img = img.transpose((2, 0, 1))
    img = img.reshape((1,) + img.shape)
    plt.figure()

    input = torch.from_numpy(img).float()
    input = torch.autograd.Variable(input).cuda()
    out = model(input).cpu().data.numpy()
    out = out.reshape(-1,2)
    raw_img = cv2.resize(raw_img,(out_size,out_size))
    plt.imshow(raw_img)
    plt.scatter((out[:, 0] * 50 + 100), out[:, 1]* 50 + 100, s=10, marker='.', c='r')
    plt.show()