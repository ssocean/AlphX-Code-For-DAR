# coding=utf-8
import argparse
import imp
import json
import math
import os
from random import random
import imutils
import os.path as osp
import sys
from time import sleep
import cv2
import numpy as np
import torch
from mmcv import Config
import glob
import torch
import numpy as np
from scipy.special import logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
from torchvision import transforms
from PIL import Image
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import AverageMeter, Corrector, ResultFormat, Visualizer
from tqdm import tqdm
# coding=utf-8
# cv2解决绘制中文乱码
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
from collections import defaultdict
H = 64
is_RGB = True
config = {
    'data_dir': 'data/mnt/ramdisk/max/90kDICT32px/',
    'img_width': 512,
    'img_height': H,
    'map_to_seq_hidden': 256,
    'rnn_hidden': 512,
    'leaky_relu': False,
}
def adapt_rotate(image,angle):
    # image = imutils.resize(image, width=300)
    # 获取图像的维度，并计算中心
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 顺时针旋转33度，并保证图像旋转后完整~,确保整个图都在视野范围
    rotated = imutils.rotate_bound(image, angle)
    # showAndWaitKey('rst',rotated)
    return rotated
def get_hor_projection(img_bin):
    rst = np.sum(img_bin,axis=1)/255
    return rst.tolist()
def is_bin_bg_white(img):
    '''_summary_
    判断二值图背景是否为白色
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    if isinstance(img, str):
        img = cv2.imread(img,0)
    elif isinstance(img, np.ndarray):
        pass
    # print(img.shape)
    assert len(img.shape)==2,'input should only have one channel'
    h,w = img.shape
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    max_val = h*w*255
    current_val = np.sum(img)
    ratio = current_val/max_val
    
    if ratio > 0.5:
        return True
    return False
def crop_by_hor_projection(hor_projection,threshold):
    l = len(hor_projection)
    
    top = 0
    down = l
    is_top_clear = False
    is_down_clear = False

    #遍历两端
    for i in range(l//2):
        if hor_projection[i]>threshold and not is_top_clear:
            top = i
            is_top_clear = True
        if hor_projection[l-1-i]>threshold and not is_down_clear:
            down = l-1-i
            is_down_clear = True
        if is_top_clear and is_down_clear:
            break

    return top,down
def otsu_bin(img: np.ndarray):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res

def getCorrect2(img,is_bin_bg_white=True):
    #读取图片，灰度化
    src = img
    bin = otsu_bin(src)
    
    if not is_bin_bg_white:
        bin = 255-bin
    # showAndWaitKey("gray",gray)
    #腐蚀、膨胀
    kernel = np.ones((5,5),np.uint8)
    erode_Img = cv2.erode(bin,kernel)
    eroDil = cv2.dilate(erode_Img,kernel)
    # showAndWaitKey("eroDil",eroDil)
    #边缘检测
    canny = cv2.Canny(eroDil,50,150)
    # showAndWaitKey("canny",canny)
    #霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 40,minLineLength=20,maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    #画出线条
    ks = []
    thetas = []
    if lines is None:
        return img
    # print(lines.shape)
    # lines.sort(key=dis_btn_points)
    max_dis = 0
    angle = 0
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line[0]
        r = pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
        k = float(y1-y2)/(x1-x2)
        theta = np.degrees(math.atan(k))
        ks.append(k)
        thetas.append(theta)
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        if r>max_dis:
            max_dis = r
            angle = theta
    # print(thetas)
    theta = -angle
    if theta == 0 or abs(theta)>60:
        return img
    """
    旋转角度大于0，则逆时针旋转，否则顺时针旋转
    """
    rotateImg = adapt_rotate(src,theta)
    

    return rotateImg # [top:down,:]
def auto_make_directory(dir_pth: str):
    '''
    自动检查dir_pth是否存在，若存在，返回真，若不存在创建该路径，并返回假
    :param dir_pth: 路径
    :return: bool
    '''
    if os.path.exists(dir_pth):  ##目录存在，返回为真
        return True
    else:
        os.makedirs(dir_pth)
        return False
def load_chars():
    CHARS = ''
    code_list_path = ''
    with open(os.path.join(code_list_path, 'codelist.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            CHARS =  CHARS + line.strip()
        return CHARS
NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01

def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob))
    # initial of beams: (empty_str, (log(1.0), log(0.0)))

    for t in range(length):
        new_beams_dict = defaultdict(lambda: (NINF, NINF))  # log(0.0) = NINF

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                # if new_prefix == prefix
                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                # if new_prefix == prefix + (c,)
                new_prefix = prefix + (c,)
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        # sorted by log(blank_prob + non_blank_prob)
        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10):
    emission_log_probs = np.transpose(log_probs.detach().cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list
class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self,in_planes, kernel_size):
        super(CBAM, self).__init__()
        # self.sa = SpatialAttention(kernel_size)
        self.ca = ChannelAttention(in_planes)

    def forward(self, x):
        x = x * self.ca(x)
        # x = x * self.sa(x)
        return x


class CRNN_CBAM(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN_CBAM, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            cnn.add_module(f'cbam{i}', CBAM(output_channel, 3))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)
def crnn_rec(crnn,image,LABEL2CHAR,tfs,device):

    crnn.to(device)
    decode_method = 'beam_search'
    decode_method = 'greedy'
    beam_size = 5
    image = np.array(image)
    # print(image.shape)
    image = tfs(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    
    logits = crnn(image)
    # print(logits.shape)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs = log_probs.detach()
    preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                        label2char=LABEL2CHAR)
    return ''.join(preds[0])


def rotateAntiClockWise90(img):  # 顺时针旋转90度
    # img = cv2.imread(img_file)
    trans_img = cv2.transpose(img)
    img90 = cv2.flip(trans_img, 0)
    # cv2.imshow("rotate", img90)
    # cv2.waitKey(0)
    return img90

def write_csv(rst: list, file_pth: str, overwrite=False):
    '''
    :param rst:形如[('val1', val2),...,('valn', valn)]的列表
    :param file_pth:输出csv的路径
    :return:
    '''
    mode = 'w+' if overwrite else 'a+'
    file = open(file_pth, mode, encoding='utf-8', newline='')

    csv_writer = csv.writer(file)

    csv_writer.writerows(rst)

    file.close()
def overlapping_seg(img):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param img_name: 待切图片名称
    :return: [子图1,子图2,...,子图N]
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    h,w,c = img.shape
    # print(h,w,c)
    patch_h = H
    ratio = patch_h/h
    img = cv2.resize(img, (int(w*ratio), patch_h))
    # print(f'overlap resized img.shape:{img.shape}')
    h = patch_h
    w = int(w*ratio)
    
    # 不要改动
    patch_w = 512
    stride_w = 256
    # 以长度 patch_h 步长stride_h的方式滑动
    patch_h = H
    stride_h = H
    # print(img.shape[1],patch_w)
    if patch_w-img.shape[1] < 30:
        img = cv2.copyMakeBorder(img,0,0,0,64,cv2.BORDER_CONSTANT,value=(0,0,0))
    if img.shape[1]<patch_w:
        rst = cv2.copyMakeBorder(img,0,0,0,patch_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
        # print(f'未达到长度，直接返回')
        return [rst]
    # print(ratio)
    # print(img.shape)
    
    # print(img.shape)
    n_w = int((w-patch_w)/stride_w)*stride_w+patch_w
    n_h = int((h-patch_h)/stride_h)*stride_h+patch_h

    img = cv2.resize(img, (n_w, n_h))
    n_patch_h = (h-patch_h)//stride_h+1
    n_patch_w = (w-patch_w)//stride_w+1
    n_patches = n_patch_h*n_patch_w
    rst = []
    for i in range(n_patch_w):
        for j in range(n_patch_h):
            y1 = j * stride_h
            y2 = y1 + patch_h
            x1 = i * stride_w
            x2 = x1 + patch_w
            roi = img[y1:y2,x1:x2]
            # print(f'roi.shape:{roi.shape}')
            rst.append(roi)
    if len(rst)==0:
        return [img]
    
    return rst


# s = ['媱怒癡性即是觧脱','晨觧脱二荅也増上𢢔者','舍也増上𢢔者未淂謂淂也身','未淂謂淂也身子𢴃小乘𫠦','身子𢴃小乘𫠦證非増上𢢔']
# overlapping_seg(r'F:\Data\GJJS-dataset\dataset\train\char_seq_bin_2\image_0_11.jpg')
def merge_str(a:str,b:str,k=2):
    if a != '':
        key = b[1:1+k]
        # print(key)
        index = a.rfind(key) #,len(a)-k-1,len(a)
        # 如果无法合并
        if index == -1:
            # print(f'unable to merge str, return the concat of {a} and {b}')
            rst = a + b #对编辑距离来说 该操作效果更好
        else:
            rst = a[:index]+b[1:]
        return rst
    else:
        return b
def merge_strs(strs:list):
    rst = ''
    
    for i in strs:
        rst = merge_str(rst,i)
    # if len(strs)>1:
    #     print(strs)
    #     print(f'multiple str merge rst:{rst}')
    return rst
def cv2_chinese_text(img, text, position, textColor=(0, 0, 255), textSize=30):
    if text is None:
        return img
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("/opt/data/private/pan_pp.pytorch/font/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle,direction='ttb')
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def points_to_poly(points):
    poly = np.array(points).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    return [poly.reshape((-1, 1, 2))]
def resize_contour(cnts,ori_size,rst_shape):
        '''
        原地操作函数，由于原图尺寸的变换将会导致标注信息的变换，该方法完成在图片尺寸变换时标注信息的同步转换。
        最好由低分辨率放大至高分辨率
        :return:
        '''
        o_h, o_w = ori_size
        r_h, r_w= rst_shape
        height_ratio = r_h / o_h
        width_ratio = r_w / o_w  # 计算出高度、宽度的放缩比例
        ratio_mat = [[width_ratio,0],[0,height_ratio]]
        # print(points_to_poly(cnts).shape)
        return (np.array(cnts).astype(np.int32).reshape((-1)).reshape((-1,  2))@ratio_mat).astype(np.int32) # n×2 矩阵乘 2×2
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para / 1e6))
    print('-' * 90)
def tensor_to_ndarray(t:torch.Tensor):
    cpu_tensor = t.cpu()
    res = cpu_tensor.detach().numpy()  # 转回numpy
    # print(res.shape)
    res = np.squeeze(res, 1)
    # res = np.swapaxes(res, 0, 2)
    # res = np.swapaxes(res, 0, 1)
    return res
def extract_roi_by_cnt(img_ori,point):
    img = img_ori.copy()
    point = point.copy()
    poly = np.array(point).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    # 定义四个顶点坐标
    pts = poly.reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(pts)  #轮廓

    # 画多边形 生成mask
    mask = np.zeros(img.shape, np.uint8)
    mask2 = cv2.fillPoly(mask.copy(), [pts],
                            (255, 255, 255))  # 用于求 ROI
    ROI = cv2.bitwise_and(mask2, img)[y:y + h, x:x + w]
    # showim(ROI)
    return ROI
def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))

def find_cnt_center(cnt):
    '''_summary_
    计算轮廓cnt的中心坐标
    Args:
        cnt (_type_): _description_

    Returns:
        _type_: _description_
    '''
    M = cv2.moments(cnt) #计算矩特征
    if M["m00"] == 0:
        return (-1,-1)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)
def merge_regions(img,cnts):
    mask = np.zeros(img.shape[:2], np.uint8)
    for c in cnts:
        poly = np.array(c).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        mask = cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))],
                                 (255, 255, 255)) 
    # cv2.imwrite('/opt/data/private/temp/ori_region.png',mask)
    h,w = img.shape[:2]
    k = 3 #11
    # print(k)
    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(mask, kernel, iterations = k)
    img_dilate = otsu_bin(img_dilate)
    cv2.imwrite('/opt/data/private/temp/img_dilate.png',img_dilate)
    contours,hierarchy = cv2.findContours(img_dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # vis = img_dilate.copy()
    # for contour in contours:
    #     poly = np.array(contour).astype(np.int32).reshape((-1))
    #     poly = poly.reshape(-1, 2)
    #     # print(poly)
    #     vis = cv2.fillPoly(np.zeros(img.shape[:2], np.uint8), [poly.reshape((-1, 1, 2))],(255, 0, 0)) 
    #     # img_dilate = cv2.polylines(img_dilate.copy(), [poly.reshape((-1, 1, 2))], True, color=(255,0, 0), thickness=5)
    #     ret = cv2.imwrite('/opt/data/private/temp/merge_region.jpg',np.zeros(img.shape[:2], np.uint8))
    return contours
def filter_inward_cnt_by_centers(cnt_centers,region_cnt):
    rst = []
    for cc in cnt_centers:
        
        # cnt = points_to_poly(cnt)
        
        # cnt = np.array(cnt).astype(np.int32).reshape((-1))
        # cnt = cnt.reshape(-1, 2)
        # print(cnt)
        # print(cc)
        cx,cy = cc[1],cc[2]
        flag = cv2.pointPolygonTest(region_cnt, (cx,cy), False)
        # print(flag)
        if flag>=0:
            # print()
            rst.append(cc)
    return rst


def order_by_y(elem):
    return elem[-1]
def order_by_x(elem):
    return elem[-2]
def order_it(img,cnts):
    cnts_dict = {}
    cnt_centers = []
    cnt_centers_wo_i = [] #without f'{i}'
    #         region_centers.append((f'{i}',cx,cy))
    #     cnt_centers_std.append((cx,cy))
    #     cnt_dicts[f'{i}'] = r_cnt
    # for cnt in cnts:
    #     cx,cy = find_cnt_center(cnt)
    for i,item in enumerate(cnts):
        cnts_dict[f'{i}'] = item
        poly = np.array(item).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cx,cy = find_cnt_center(poly)
        if cx!=-1:
            cnts_dict[f'{i}'] = poly
            cnt_centers.append((f'{i}',cx,cy))
            cnt_centers_wo_i.append((cx,cy))
        
    region_contours = merge_regions(img,cnts)
    region_centers = []
    region_centers_std = []
    region_cnt_dicts={}
    for i,r_cnt in enumerate(region_contours):
        cx,cy = find_cnt_center(r_cnt)
        if cx!=-1:
            region_centers.append((f'{i}',cx,cy))
            region_centers_std.append((cx,cy))
            region_cnt_dicts[f'{i}'] = r_cnt

    h,w = img.shape[:2]
    # mean_size = np.mean(region_centers,axis=0)
    print(f'一共有{len(region_centers)}个regions')
    
    if len(region_centers) > 1:
        # print(region_centers)
        std_size = np.std(region_centers_std, axis=0)
        # print(std_size)
        if std_size[1] > h//6: # 高度波动大，纵向排列
            #大区域根据纵坐标排序
            print('高度波动大，纵向排列')
            region_centers.sort(key=order_by_y)
        else: #std_size[1] < h//4:# 高度波动大 横向排列
            # 
            print('高度波动小，横向排列')
            region_centers.sort(key=order_by_x,reverse=True)

    rst_cnts = []
    
    for region_center in region_centers:
        # print(region_center[1:])
        region_cnt = region_cnt_dicts[f'{region_center[0]}']
        # print(region_cnt)
        in_region_ccs = filter_inward_cnt_by_centers(cnt_centers,region_cnt) #find in-region counter centers
        # print(f'in_region_ccs_len{len(in_region_ccs)}')
        # print(in_region_ccs)
        in_region_ccs.sort(key=order_by_x,reverse=True)
        # print(in_region_ccs)
        for item in in_region_ccs:#(i,x,y)
            rst_cnts.append(cnts_dict[f'{item[0]}'])
    return rst_cnts
def test(test_loader, model, cfg):
    CHARS = load_chars()  # 13980
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    num_class = len(LABEL2CHAR) + 1
    img_height = config['img_height']
    img_width = config['img_width']
    # CRNN---------------------------------------------------------
    # crnn = CRNN_CBAM(3, 64, 512, 13981,256,  512, False)
    # crnn = CRNN(1, img_height, img_width, num_class,
    #     map_to_seq_hidden=config['map_to_seq_hidden'],
    #     rnn_hidden=config['rnn_hidden'],
    #     leaky_relu=config['leaky_relu'])
    
    crnn = CRNN_CBAM(3, H, 512, 13981, 256, 512, False)
    reload_checkpoint = 'weights/64_512_best_model.pt'#'/opt/data/private/crnn-pytorch/crnn-pytorch/checkpoints/best_model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    tfs = transforms.Compose([
        iaa.Sequential([
            iaa.Resize({"height": img_height, "width": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(height=img_height, width=img_width, position="center")
        ]).augment_image,
        transforms.ToTensor()
    ])
    crnn.eval()
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)

    if cfg.vis:
        vis = Visualizer(vis_path=osp.join('vis/', cfg.data.test.type))

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_post_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500))

    print('Start testing %d images' % len(test_loader))
    result = []
    for idx, data in enumerate(tqdm(test_loader)):
        # print(f'Testing {idx}/{len(test_loader)}\r', end='', flush=True)
        # sleep(0.1)
        # prepare input
        data['imgs'] = data['imgs'].cuda()
        img_pth = data['img_metas']['img_path'][0]
        img_name = data['img_metas']['img_name'][0]
        # print(data['img_metas'])
        data.update(dict(cfg=cfg))
        filename, file_extension = os.path.splitext(img_name)
        result_name='res_%s.jpg' % filename
        print(result_name)
        # print('data length:'+str(len(data)))
        # print(data)
        # forward
        with torch.no_grad():
            outputs = model(**data)
        # print(outputs['bboxes'])
        # print(outputs['words'])
        
        bboxes = outputs['bboxes']
        #img_name='image_105.jpg'#os.path.basename(image_path)
        ori_img_fullsize = cv2.imread(img_pth,1)#r'/opt/data/private/pan_pp.pytorch/data/test/image_105.jpg'
        ori_img_for_roi = ori_img_fullsize.copy()
        ori_h,ori_w = ori_img_fullsize.shape[:2]
        ori_img = cv2.resize(ori_img_fullsize, dsize=(896, 896))# 改变后尺寸
        # print(f'bbox num before order:{len(bboxes)}')
        ordered_bboxes = order_it(ori_img.copy(),bboxes)
        # print(f'bbox num after order:{len(ordered_bboxes)}')
        #画图
        # for i, bbox in enumerate(bboxes):
        #     poly = np.array(bbox).astype(np.int32).reshape((-1))
        #     poly = poly.reshape(-1, 2)
        #     # cx,cy = find_cnt_center(poly)
            
        #     cv2.polylines(ori_img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)
        #     ori_img = cv2_chinese_text(ori_img, outputs['words'][i], outputs['bboxes'][i][:2], textColor=(0, 255, 0), textSize=30)
            # cv2.circle(ori_img, (cx,cy), 3, (255, 0, 0), -1)
        # is_bg_white = is_bin_bg_white(CRNN_ROI_bin[cy-h//4:cy+h//4,cx-w//4:cy+w//4])

        
        with torch.no_grad():
            result = []
            for i, bbox in enumerate(ordered_bboxes):
                # if 'words' in outputs.keys():
                #     ori_img = cv2_chinese_text(ori_img, outputs['words'][i], outputs['bboxes'][i][:2], textColor=(0, 255, 0), textSize=30)
                # print(ori_img.shape)
                cnts = resize_contour(bbox.copy(),(896,896),(ori_h,ori_w))
                
                rst = np.round(cnts.reshape((-1))).tolist() # 点的坐标
                if len(rst)<=6:
                    continue
                # print(cnts) #ndarray
                CRNN_ROI = extract_roi_by_cnt(ori_img_for_roi,cnts)
                # print(CRNN_ROI.shape)
                if CRNN_ROI is None:
                    continue
                
                CRNN_ROI = rotateAntiClockWise90(CRNN_ROI)
                # print(CRNN_ROI.shape)
                
                # h,w = img_bin.shape[:2]
                # cx,cy = (w//2,h//2)
                # is_bin_bg_white(img_bin)   
                CRNN_ROI_bin = cv2.cvtColor(CRNN_ROI.copy(), cv2.COLOR_BGR2GRAY)
                _, CRNN_ROI_bin = cv2.threshold(CRNN_ROI_bin, 10, 255, cv2.THRESH_BINARY)
                h,w = CRNN_ROI_bin.shape[:2]
                cx,cy = (w//2,h//2)        
                CRNN_ROI = getCorrect2(CRNN_ROI,is_bin_bg_white(CRNN_ROI_bin[cy-h//4:cy+h//4,cx-w//4:cy+w//4]))#[cy-h//4:cy+h//4,cx-w//6:cy+w//6]
                # print(CRNN_ROI.shape)
                check_bin_img = cv2.cvtColor(CRNN_ROI.copy(), cv2.COLOR_BGR2GRAY)
                _, rotateImg_bin = cv2.threshold(check_bin_img, 10, 255, cv2.THRESH_BINARY)
                h,w = check_bin_img.shape[:2]
                cx,cy = (w//2,h//2)

                threshold,_ = rotateImg_bin.shape[:2]    
                # showAndWaitKey('rotateImg_bin',rotateImg_bin)
                hor_proj = get_hor_projection(rotateImg_bin)
                top,down = crop_by_hor_projection(hor_proj,threshold//20)

                # print(top,down)
                CRNN_ROI = CRNN_ROI[top:down,:]
                # patches是以长度256 步长128的方式在32*N (N>1.5*256)的图像上切出来的
                if is_RGB:
                    CRNN_ROI = cv2.cvtColor(CRNN_ROI.copy(), cv2.COLOR_BGR2RGB)
                patches = overlapping_seg(CRNN_ROI)
                rec_seq = []
                # print(f'len(patches):{len(patches)}')
                for j,patch_cv in enumerate(patches):
                    # print(f'patch尺寸:{patch_cv.shape}')
                    if is_RGB:
                        patch = Image.fromarray(patch_cv)#.convert('L')
                    else:
                        patch = Image.fromarray(patch_cv).convert('L')
                    # 获得序列识别结果识别结果
                    
                    output = crnn_rec(crnn,patch,LABEL2CHAR,tfs,device)
                    # print(f'crnn输出：{output}')
                    rec_seq.append(output)
                    # seq_vis_dir = '/opt/data/private/pan_pp.pytorch/outputs/seq-crnn-r50-gray-vis'
                    # auto_make_directory(seq_vis_dir)
                    # if i<=5:
                    #     cv2.imwrite(f'{seq_vis_dir}/{filename}_{i}_{j}_{output}.png',patch_cv)
                # print(f'strs waiting for being merged:{rec_seq}')
                # 字符串合并算法
                rec_rst = merge_strs(rec_seq)


                # print(outputs['bboxes'][i])
                # print(f'rec_rst:{rec_rst}')
                # cnts = outputs['bboxes'][i]
                # print(bbox)
                
                # print(cnts)
                
                
                # .append(rec_rst)
                rst.append(rec_rst)
                if rec_rst == '' or len(rst)<=7:
                    continue
                # print(rst)
                result.append(tuple(rst))
                # print(outputs['words'][i])
                poly = np.array(cnts.copy()).astype(np.int32).reshape((-1))
                poly = poly.reshape(-1, 2)
                # cx,cy = find_cnt_center(poly)
                cv2.polylines(ori_img_fullsize, [poly.reshape((-1, 1, 2))], True, color=(0, 190, 0), thickness=1)
                # print(outputs['bboxes'][i][:2])
                ori_img_fullsize = cv2_chinese_text(ori_img_fullsize, f'{i},'+rec_rst, cnts[0], textColor=(255, 0, 0), textSize=20)
            csv_dir = '/opt/data/private/csv-64-512-CBAM-0928'
            auto_make_directory(csv_dir)
            write_csv(result, file_pth=f'{csv_dir}/{filename}.csv', overwrite=True)
        out_dir = '/opt/data/private/pan_pp.pytorch/outputs/det-r50-929'
        auto_make_directory(out_dir)
        cv2.imwrite(os.path.join(out_dir ,result_name), ori_img_fullsize)
        if cfg.report_speed:
            report_speed(outputs, speed_meters)
        # post process of recognition
        if with_rec:
            # continue
            outputs = pp.process(data['img_metas'], outputs)
        
        # save result
        rf.write_result(data['img_metas'], outputs)

        # visualize
        if cfg.vis:
            vis.process(data['img_metas'], outputs)
        # sleep(2)
        
    print('Done!')

def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    cfg.update(dict(vis=args.vis))
    cfg.update(dict(debug=args.debug))
    cfg.data.test.update(dict(debug=args.debug))
    print(json.dumps(cfg._cfg_dict, indent=4))
    # *******************************************************************
    # Char2ID
    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    # *******************************************************************
    # model
    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(
            dict(
                voc=data_loader.voc,
                char2id=data_loader.char2id,
                id2char=data_loader.id2char,
            ))
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(
                args.checkpoint))

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model_structure(model)
    # test
    test(test_loader, model, cfg)


'''
python3 test.py config/pan_pp/pan_pp_r18_DAR_det_only.py checkpoints/pan_pp_r18_DAR_det_only/checkpoint.pth.tar
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.config = '/opt/data/private/pan_pp.pytorch/config/pan_pp/pan_pp_r18_DAR_det_only.py'#'config/pan_pp/pan_pp_r18_DAR_det_only-r101.py'
    # checkpoint_all_full+180ep.pth.tar
    # checkpoint_all_903_100ep.pth.tar
    # checkpoint.pth.tar
    # checkpoint105.903_1ep_noaug.pth.tar
    args.checkpoint = 'weights/checkpoint.pth.tar'#'checkpoints/pan_pp_r18_DAR_det_only/checkpoint.pth.tar' #/opt/data/private/pan_pp.pytorch/checkpoints/pan_pp_r18_DAR_det_only-r101/checkpoint_230ep.pth.tar
    main(args)
