from math import tanh
from random import random

from torch import nn
from torch.nn.functional import softmax
from torchvision import models


import argparse
from os.path import splitext

import cv2
import torch
import numpy as np

from tqdm import tqdm


import os
import glob



parser = argparse.ArgumentParser()
parser.add_argument("-imgs_dir")
parser.add_argument("-out_dir")
parser.add_argument("-model_pth")
args = parser.parse_args()

imgs_dir = args.imgs_dir
out_dir = args.out_dir
def get_mask_by_color(img_pth,r=183,g=209,b=235):
    clr=[b,g,r]
    minClr = np.array(clr)-1
    maxClr = np.array(clr)+1
    img = cv2.imread(img_pth)
    # hsv = cv2.cvtColor(img,)
    mask = cv2.inRange(img,minClr,maxClr)
    # print(mask.shape)
    # showim(mask)
    return mask
if __name__ == '__main__':


    imgs_dir =r'/opt/data/private/pan_pp.pytorch/data/image'
    out_dir = r'output/'
    args.model_pth = r'/opt/data/private/pan_pp.pytorch/checkpoints/pan_pp_r18_DAR/checkpoint.pth.tar'


    resnet = models.resnet50()
    del resnet.fc
    # 换一个新的全连接层
    resnet.add_module('fc', nn.Linear(2048, 8))



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet.load_state_dict(torch.load(args.model_pth))
    resnet.to(device)
    resnet.eval()

    imgs_list = get_files_pth(imgs_dir)
    rst_list = []
    i=0
    for pth in tqdm(imgs_list):
        i+=1
        mask = get_mask_by_color(pth)
        num = np.sum(mask)
        img_name = get_filename_from_pth(pth)
        if (num/255)/(1080*2400)>0.0001:#如果探测到大量超出颜色
            print(f'6类类别错误：{tanh(900*(num/255)/(1080*2400))}')
            rst_list.append((img_name,tanh(900*(num/255)/(1080*2400))))
        else: 
            with torch.no_grad():
                img_np = cv2.imread(pth)
                img_np = cv2.resize(img_np,(240,320))
                img_tensor = torch.from_numpy(img_np)  # 转tensor
                img_tensor = img_tensor.unsqueeze(0).transpose(3, 1).transpose(2, 3)  # 转维度


                img_tensor = img_tensor.to(device=device, dtype=torch.float32)  # 转设备、类型
                # print(img_tensor.shape)
                pred = resnet(img_tensor)

                pred_prob = softmax(pred)

                pred_label = torch.argmax(pred_prob[0],0)

                # print(pred_prob)
                #     print(f'{pth},{pred_label},{pred_prob[0][pred_label]}')

                rst_list.append((get_filename_from_pth(pth),float(1 - pred_prob[0][pred_label]))) #pred_prob[0][0] 分类正确的概率


                    # if i > 50:
                    #     break

    print(rst_list)
