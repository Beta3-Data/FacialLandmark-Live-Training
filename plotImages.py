import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation

from models.basenet import SqueezeNet
import torch

import glob

filenames = glob.glob("dataset/dd/test/*.jpg")

fig, ax = plt.subplots()

nx = 224
ny = 224
sc = ax.scatter(nx,ny, s=10, marker='.', c='r')
data = np.zeros((nx, ny))
im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)

def landmark_prediction(i):
    out_size = 224
    model = load_model()
    model = model.eval()
    img = cv2.imread(filenames[i])
    img = cv2.resize(img,(224,224))
    raw_img = img
    img = img/255.0
    img = img.transpose((2, 0, 1))
    img = img.reshape((1,) + img.shape)
    input = torch.from_numpy(img).float()
    input = torch.autograd.Variable(input).cuda()
    out = model(input).cpu().data.numpy()
    out = out.reshape(-1,2)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    raw_img = cv2.resize(raw_img,(out_size,out_size))
    sc.set_offsets(np.c_[(out[:, 0] * 50 + 100),out[:, 1]* 50 + 100])
    im.set_data(raw_img)
    return im

def init():
    landmark_prediction(0)

def animate(x):
    landmark_prediction(x)


def load_model():
    model = SqueezeNet(136).cuda()
    checkpoint = torch.load("checkpoint/1011/facelandmark_squeezenet_64_67.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    return model

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny)

plt.show()