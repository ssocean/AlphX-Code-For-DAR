# coding=utf-8
import os
import sys
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse
import csv
import glob
import imp
import json
import math
import os
import os.path as osp
import sys
from collections import defaultdict
from random import random
from time import sleep

import cv2
import imgaug.augmenters as iaa
import imutils
import numpy as np
import torch
# coding=utf-8
# cv2解决绘制中文乱码
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mmcv import Config
from PIL import Image, ImageDraw, ImageFont
from scipy import signal
from scipy.io import wavfile
from scipy.special import \
    logsumexp  # log(p1 + p2) = logsumexp([log_p1, log_p2])
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import AverageMeter, Corrector, ResultFormat, Visualizer
H = 64
is_RGB = True


def ndarray_to_tensor(ndarray:np.ndarray):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.Tensor(ndarray)
    # t.to(device)
    return t


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
    img_bin=img_bin
    # showim(img_bin)
    rst = np.sum(img_bin,axis=1)//255
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


def get_white_ratio(bbox:np.ndarray):
    '''
    针对黑底白字
    '''
    if len(bbox.shape)>2:
        #三通道 转灰度图
        bbox_gray = cv2.cvtColor(bbox,cv2.COLOR_BGR2GRAY)
    else:
        bbox_gray = bbox

    _,bbox_bin = cv2.threshold(bbox_gray,1,255,cv2.THRESH_BINARY)
    bbox_bin.astype(np.uint16)
    h,w = bbox_bin.shape[:2]

    bbox_bin = bbox_bin/255
    current_val = np.sum(bbox_bin)
    ratio = current_val/(h*w) #
    return ratio


def get_white_ratio_cuda(bbox:np.ndarray):
    '''
    输入图像应为单通道,cuda加速版本
    针对黑底白字
    '''
    if len(bbox.shape)>2:
        #三通道 转灰度图
        bbox_gray = bbox[:,:,0]
    else:
        bbox_gray = bbox

    _,bbox_bin = cv2.threshold(bbox_gray,1,255,cv2.THRESH_BINARY)
    bbox_tensor = ndarray_to_tensor(bbox_bin)
    # bbox_bin.astype(np.uint16)
    h,w = bbox_tensor.shape[:2]

    bbox_tensor = bbox_tensor/255
    # current_val = np.sum(bbox_bin)
    ratio = bbox_tensor.sum()/(h*w) #
    return ratio


def is_img_bg_black(img:np.ndarray):
    '''
    该函数默认img为896*896大小 PIL RGB

    '''

    img = cv2.resize(img,(896,896))
    img = img[300:600,300:600]
    img = img.astype(np.uint16)
    r,g,b = cv2.split(img)
    # 如果是灰度图

    if np.sum(b) == np.sum(g) == np.sum(r):
        image = b
        image = image[image<5]
        black_pix_num = len(image)
        # print(black_pix_num/802816)
        if (black_pix_num/(300*300))>0.5:
            return True
    return False


def crop_by_hor_projection(hor_projection,threshold):
    '''_summary_
    根据投影信息返回两端第一次非零元素出现位置
    Args:
        hor_projection (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_ top / down
    '''
    l = len(hor_projection)
    top = 0
    down = l
    is_top_clear = False
    is_down_clear = False
    # print(f'threshold is {threshold}')
    # print(hor_projection[-5:])
    #遍历两端
    threshold = 0
    for i in range(l):
        if hor_projection[i]>threshold and not is_top_clear:
            top = i
            is_top_clear = True
        if hor_projection[l-1-i]>threshold and not is_down_clear:
            down = l-1-i
            is_down_clear = True
        if is_top_clear and is_down_clear:
            break
    # print(f'{top,down}/{l}')
    return top,down


def otsu_bin(img: np.ndarray):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def getCorrect1(img):
    '''_summary_
    霍夫变换 要求输入图像为单通道图像
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    #读取图片，灰度化
    src = img

    _,bin = cv2.threshold(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

    canny = cv2.Canny(bin,50,150)
    h,w = img.shape[:2]
    min_len = min(max(h, w) // 6, 30)
    max_gap = max(min(h, w) // 6, 50)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 60, minLineLength=min_len, maxLineGap=max_gap)

    if lines is None:
        return img

    max_dis = 0
    angle = 0
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line[0]# [[line]]
        r = pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
        k = float(y1-y2)/(x1-x2)
        theta = np.degrees(math.atan(k))
        if r>max_dis:
            max_dis = r
            angle = theta
    theta = -angle

    if theta == 0 or abs(theta)>60:
        return img

    rotateImg = adapt_rotate(src,theta)

    rotateImg_gray = cv2.cvtColor(rotateImg,cv2.COLOR_BGR2GRAY)
    _,rotateImg_bin = cv2.threshold(rotateImg_gray, 1, 255, cv2.THRESH_BINARY)

    threshold,_ = rotateImg_bin.shape[:2]
    hor_proj = get_hor_projection(rotateImg_bin)
    top,down = crop_by_hor_projection(hor_proj,threshold//20)

    return rotateImg [top:down,:]


def getCorrect2(img):
    '''_summary_
    基于轮廓的对齐，可用于矫正任意弯曲的图像
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    h,w = img.shape[:2]
    if len(img.shape)==3:
        rst = np.zeros([h,w,3],dtype=np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _,img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    else:
        rst = np.zeros([h,w],dtype=np.uint8)
        img_gray = img
        _,img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    none_zero_index = (img_bin!=0).argmax(axis=0)
    for i,indent in enumerate(none_zero_index):
        if len(img.shape)==3:
            rst[:,i,:] = np.roll(img[:,i,:], -indent,axis=0)
        else:
            rst[:,i] = np.roll(img[:,i], -indent)
    if len(img.shape)==3:
        deskew_gray_rst = cv2.cvtColor(rst, cv2.COLOR_BGR2GRAY)
    else:
        deskew_gray_rst = rst
    _, deskew_bin_rst = cv2.threshold(deskew_gray_rst, 1, 255, cv2.THRESH_BINARY)
    # showim(deskew_rst)
    hor_proj = get_hor_projection(deskew_bin_rst)
    threshold,_ = deskew_bin_rst.shape[:2]
    top,down = crop_by_hor_projection(hor_proj,threshold//20)
    rst = rst[top:down,:]
    return rst


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


def rotateAntiClockWise90(img):  # 顺时针旋转90度
    # img = cv2.imread(img_file)
    trans_img = cv2.transpose(img)
    img90 = cv2.flip(trans_img, 0)
    # cv2.imshow("rotate", img90)
    # cv2.waitKey(0)
    return img90


def deskew(CRNN_ROI):
    deskew_rst = CRNN_ROI
    # 根据比例决定是否进行扭曲矫正，resize操作用于减少计算量
    deskew_rst_for_ratio = cv2.resize(deskew_rst,(100,32))
    if get_white_ratio_cuda(deskew_rst_for_ratio)<0.85:
        deskew_rst = getCorrect1(deskew_rst)
    deskew_rst_for_ratio = cv2.resize(deskew_rst,(100,32))
    if get_white_ratio_cuda(deskew_rst)<0.85:
        deskew_rst = getCorrect2(deskew_rst)
    return deskew_rst


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
    # print(f'ori input img shape:{img.shape}')
    h,w = img.shape[:2]
    # print(h,w,c)
    patch_h = H
    ratio = patch_h/h
    resized_w = int(w*ratio)
    img = cv2.resize(img, (resized_w, patch_h))
    # print(f'img.shape waiting for overlap resized :{img.shape}')
    h = patch_h

    patch_w = 512

    stride_w = 256

    # 以长度 patch_h 步长stride_h的方式滑动
    stride_h = H
    # print(img.shape[1],patch_w)

    if patch_w>img.shape[1] and patch_w-img.shape[1] < 30:
        rst = cv2.copyMakeBorder(img,0,0,0,64,cv2.BORDER_CONSTANT,value=(0,0,0))
        rst= cv2.resize(rst,(patch_w,H))
        # print(f'未达到长度-30，直接返回。返回形状:{rst.shape}')
        return [rst]
    if img.shape[1]<patch_w:
        rst = cv2.copyMakeBorder(img,0,0,0,patch_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
        # print(f'未达到长度，直接返回。返回形状:{rst.shape}')
        return [rst]
    # print(ratio)
    # print(img.shape)

    # print(f'after copymakeborder img shpae:{img.shape}')
    rescaled_h,rescaled_w = img.shape[:2]
    n_w = int(math.ceil((rescaled_w-patch_w)/stride_w))*stride_w+patch_w
    n_h = H

    img = cv2.copyMakeBorder(img,0,0,0,n_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
    # img = cv2.resize(img, (n_w, n_h))

    # print(f'长边自适应尺寸:{img.shape}')

    rescaled_h,rescaled_w = img.shape[:2]
    n_patch_h = (rescaled_h-patch_h)//stride_h+1
    assert n_patch_h==1,'n_patch_h!=1'
    n_patch_w = (rescaled_w-patch_w)//stride_w+1

    # print(f'n_patch_h：{n_patch_h}，n_patch_w：{n_patch_w}')
    rst = []
    for i in range(n_patch_w):
        x1 = i * stride_w
        x2 = x1 + patch_w
        roi = img[0:H,x1:x2]
        # print(f'roi.shape:{roi.shape}')
        rst.append(roi)
    if len(rst)==0:
        print('overlap len is 0, this means something could be wrong but not that so lethel')
        return [img]

    return rst


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
    return rst

def cv2_chinese_text(img, text, position, textColor=(0, 0, 255), textSize=30):
    if text is None:
        return img
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
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
    inner_pts = pts - np.array([x,y])
    # print(pts)
    # 画多边形 生成mask
    img_patch = img[y:y + h, x:x + w]
    mask = np.zeros(img.shape, np.uint8)[y:y + h, x:x + w]
    mask2 = cv2.drawContours(mask.copy(), [inner_pts], -1, (255,255,255), thickness=-1)
    ones = 2*np.ones(img_patch.shape,dtype=np.uint8)
    img_patch = cv2.add(img_patch,ones)
    ROI = cv2.bitwise_and(mask2, img_patch)
    ROI = cv2.rotate(ROI, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
    if len(cnt)<=6:
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
    # cv2.imwrite('temp/ori_region.png',mask)
    # cnt_count = len(cnts)
    # h,w = img.shape[:2]
    k = 3 #11
    # print(k)
    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(mask, kernel, iterations = k)
    img_dilate = otsu_bin(img_dilate)
    # cv2.imwrite('temp/img_dilate.png',img_dilate)
    contours,hierarchy = cv2.findContours(img_dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return contours

def sort_region(img, cnts, model, device,writer,idx):
    # print(f'img_shape:{img.shape}')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shrinked_imgs = cv2.resize(img, (256, 256))
    shrinked_polys = []
    for id, poly in enumerate(cnts):
        poly = resize_contour(poly, (896,896), (256, 256))
        shrinked_polys.append(poly)
    shrinked_cnts = np.array(shrinked_polys)
    shrinked_mask = np.zeros(shrinked_imgs.shape[:2], dtype=np.uint8)

    input = cv2.drawContours(shrinked_mask, shrinked_cnts, -1, 1, thickness=-1)
    img_tensor = torch.from_numpy(input)  # 转tensor
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)  # 转设备、类型
    # writer.add_images('order-input', img_tensor, global_step=idx, dataformats='NCHW')
    mask_pred = model(img_tensor)
    # GLOBAL_ORDER_ID =  GLOBAL_ORDER_ID + 1
    # writer.add_images('order', mask_pred, global_step=idx, dataformats='NCHW')
    pred_np = mask_pred.cpu().detach().numpy()  # 转回numpy
    rst = np.squeeze(pred_np, 0).swapaxes(0, 2).swapaxes(0, 1)

    rst = rst.astype(np.float64).squeeze(-1)
    # showim(rst)

    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv2.dilate(shrinked_mask.copy(), kernel, iterations=1)
    img_dilate = cv2.erode(img_dilate.copy(), kernel, iterations=1) #255
    merged_contours, _ = cv2.findContours(img_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #256*256


    merged_contour_ids = {}
    merged_contour_values = {}
    for i in range(len(merged_contours)):
        merged_contour_ids.update({i:merged_contours[i]})
        blank = np.zeros((shrinked_imgs.shape[0], shrinked_imgs.shape[1]), dtype=np.uint8)
        cv2.drawContours(blank, merged_contours, i, 255, -1)
        # showim(blank, 'blank', False)
        mean_val = cv2.mean(rst, blank)[0]
        # print(f'第{i}个区域均值{mean_val}')
        merged_contour_values.update({i: mean_val})
    #     cv2.drawContours(merged_mean_mask, merged_contours, i, mean_val, -1)
    # showim(merged_mean_mask)
    sorted_polys = sorted(merged_contour_values.items(), key=lambda s: s[1])
    ordered_bbox = []
    order_correspongding = {}
    for i, item in enumerate(sorted_polys):
        id = item[0]
        cnt = merged_contour_ids[id]
        cnt = resize_contour(cnt,(256, 256),(896,896))
        ordered_bbox.append(cnt)
        order_correspongding.update({i: id})
    # print(order_correspongding)
    # print(ordered_bbox) #通find contour类型一致
    return ordered_bbox

def filter_inward_cnt_by_centers(cnt_centers,region_cnt):
    rst = []
    for cc in cnt_centers:

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
    '''_summary_
    一种完全基于启发式算法的区域排序实现
    Args:
        img (_type_): _description_
        cnts (_type_): _description_

    Returns:
        _type_: _description_
    '''
    cnts_dict = {}
    cnt_centers = []
    cnt_centers_wo_i = []
    for i,item in enumerate(cnts):
        cnts_dict[f'{i}'] = item
        poly = np.array(item).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cx,cy = find_cnt_center(poly)
        if cx!=-1:
            cnts_dict[f'{i}'] = poly
            cnt_centers.append((f'{i}',cx,cy))
            cnt_centers_wo_i.append((cx,cy))

    # 获得区域中心点位置
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
    # print(f'一共有{len(region_centers)}个regions')

    if len(region_centers) > 1:
        # print(region_centers)
        std_size = np.std(region_centers_std, axis=0)
        # print(std_size)
        if std_size[1] > h//6: # 高度波动大，纵向排列
            #大区域根据纵坐标排序
            # print('高度波动大，纵向排列')
            region_centers.sort(key=order_by_y)
        else: #std_size[1] < h//4:# 高度波动大 横向排列
            #
            # print('高度波动小，横向排列')
            region_centers.sort(key=order_by_x,reverse=True)

    rst_cnts = []

    for region_center in region_centers:
        # print(region_center[1:])
        region_cnt = region_cnt_dicts[f'{region_center[0]}']
        print(region_cnt)
        # print(region_cnt)
        in_region_ccs = filter_inward_cnt_by_centers(cnt_centers,region_cnt) #find in-region counter centers
        # print(f'in_region_ccs_len{len(in_region_ccs)}')
        # print(in_region_ccs)
        in_region_ccs.sort(key=order_by_x,reverse=True)
        # print(in_region_ccs)
        for item in in_region_ccs:#(i,x,y)
            rst_cnts.append(cnts_dict[f'{item[0]}'])
    return rst_cnts


def order_it_by_unet(img,cnts,model, device,writer,idx):
    '''
    基于unet的区域排序
    '''
    cnts_dict = {}
    cnt_centers = []
    cnt_centers_wo_i = []
    for i,item in enumerate(cnts):
        cnts_dict[f'{i}'] = item
        poly = np.array(item).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cx,cy = find_cnt_center(poly)
        if cx!=-1:
            cnts_dict[f'{i}'] = poly
            cnt_centers.append((f'{i}',cx,cy))
            cnt_centers_wo_i.append((cx,cy))

    region_cnts = sort_region(img, cnts,model, device,writer,idx)
    vis = img.copy()
    for i in range(len(region_cnts)):
        cv2.drawContours(vis,region_cnts,i,(20*i,20*i,20*i),-1)

    # writer.add_images('region_cnt_vis', vis, global_step=idx, dataformats='HWC')
    # cv2.imwrite(f'outputs/order-vis/{str(uuid.uuid4())}.png',vis)
    # print(region_cnts)
    rst_cnts = []

    for region_cnt in region_cnts:
        in_region_ccs = filter_inward_cnt_by_centers(cnt_centers,region_cnt) #find in-region counter centers
        in_region_ccs.sort(key=order_by_x,reverse=True)
        for item in in_region_ccs:#(i,x,y)
            rst_cnts.append(cnts_dict[f'{item[0]}'])
    return rst_cnts


#---------------------------------------------------深度学习相关-------------------------------------------------------------

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
    '''
    模型文件，理论上可更换任意模型在此
    '''
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
    '''
    加入CBAM模块的CRNN
    '''
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



class PatchDataset(Dataset):
    def __init__(self, all_patches, tfs, opt):
        self.opt = opt
        self.all_patches = all_patches
        self.tfs = tfs
        self.nSamples = len(self.all_patches)
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        #(id,img)
        # try:

        image = np.array(self.all_patches[index][1])
        # print(image.shape)
        if is_RGB:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image).convert('L')
        # image = image[0]


        # image = torch.unsqueeze(image, 0)*

        # if self.opt.rgb:
        #     # Image.fromarray(patch_cv).convert('RGB')
        #     img = Image.fromarray(image).convert('RGB')  # for color image
        # else:

        image = np.array(image)
        image = tfs(image)


        return image


def load_chars():
    '''
    由于多人开发，未避免出错保留了两套加载字典的代码和字库
    '''
    CHARS = ''
    code_list_path = ''
    with open(os.path.join(code_list_path, 'codelist.txt'), 'r') as f:
        for i, line in enumerate(f.readlines()):
            CHARS =  CHARS + line.strip()
        return CHARS
config = {
    'data_dir': 'data/mnt/ramdisk/max/90kDICT32px/',
    'img_width': 512,
    'img_height': H,
    'map_to_seq_hidden': 128,
    'rnn_hidden': 256,
    'leaky_relu': False,
}
class Arg():
    def __init__(self):
        self.imgH = H
        self.imgW = 512
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.batch_max_length = 25
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.PAD = True
        self.sensitive = False
        self.rgb = is_RGB
        self.saved_model ='/weights-bk/best_norm_ED.pth'
        self.workers = 0
        self.batch_size = 2
        self.character = load_chars()
        self.num_class = len(self.character)
        self.num_gpu = 1
opt = Arg()
tfs = transforms.Compose([
    iaa.Sequential([
        iaa.Resize({"height": opt.imgH, "width": "keep-aspect-ratio"}),
        iaa.PadToFixedSize(height=opt.imgH, width=opt.imgW, position="center")
    ]).augment_image,
    transforms.ToTensor()
])


def crnn_rec(crnn,image,LABEL2CHAR,tfs,device):

    crnn.to(device)
    decode_method = 'beam_search'
    decode_method = 'greedy'
    beam_size = 5
    image = np.array(image)
    image = tfs(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    logits = crnn(image)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs = log_probs.detach()
    preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                        label2char=LABEL2CHAR)
    return ''.join(preds[0])

CHARS = load_chars()  # 13980
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

def seq_rec(rec_model,demo_loader,device):
    """ model configuration """
    rst = []
    # i = 0
    for image_tensors in demo_loader:
        batch_size = image_tensors.size(0)

        image = image_tensors.to(device)
        rec_model.to(device)
        # 贪心搜索
        beam_size = 1
        decode_method = 'greedy'
        # print(image.shape)
        logitss = rec_model(image)

        logitss = logitss.transpose(0,1)
        # print(logitss.shape)
        for logits in logitss:
            # print(i)
            logits=logits.unsqueeze(1)
            # print(logits.shape)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            log_probs = log_probs.detach()
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                                label2char=LABEL2CHAR)
            # print()
            rst.append(''.join(preds[0]))
            # i=i+1
    return rst


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    '''
    一个简单的UNet结构，训练文件可参考https://github.com/ssocean/Attention-U-Net
    '''
    def __init__(self, n_classes=1, n_channels=3, bilinear=True):
        self.name = 'UNet'
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = self.sigmoid(logits)
        return logits
