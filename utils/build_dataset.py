import csv
from ntpath import join
import cv2
import copy
import os
import cv2
import numpy as np
import json
from labelme import utils
import random
from tqdm import tqdm
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# workshop参见https://github.com/ssocean/workshop
from workshop.GeneralTools.FileOperator import *
from workshop.CVTools.CVTools import *


'组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里'
rst_dir = r''#'结果存放路径'
img_dir = r''#'官方图片存放路径，形如F:\dataset\train\image'
json_pth = r''#'官方json文件路径，形如F:\dataset\train\label.json'
auto_make_directory(rst_dir)
def split_list(x: list, n: int, newList=[]):
    '''_summary_
    将x拆分为多个长度为n的子list
    Args:
        x (_type_): 待拆分list
        n (_type_): 子list长度
        newList (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    '''
    if len(x) <= n:
        newList.append(x)
        return newList
    else:
        newList.append(x[:n])
        return split_list(x[n:], n)



def write_csv(rst: list, file_pth: str, overwrite=True):
    '''
    :param rst:形如[('文件名1', 概率值1),...,('文件名n', 概率值n)]的列表
    :param file_pth:输出csv的路径
    :return:
    '''
    mode = 'w+' if overwrite else 'a+'
    file = open(file_pth, mode, encoding='utf16', newline='')

    csv_writer = csv.writer(file)
    #插入标题栏 否则报4000行错误
    # rst.insert(0,('imagename', 'defect_prob'))
    csv_writer.writerows(rst)

    file.close()


class Label:
    '''
    Labelme数据对象处理类
    '''

    def __init__(self,
                 prefix: str,
                 img_name: str,
                 version: str = '4.5.7',
                 flags: dict = {},
                 shapes: list = [],
                 imagePath: str = '',
                 imageData: str = None,
                 imageHeight: int = 0,
                 imageWidth: int = 0):
        '''
        该类可由两种方法初始化，直接加载json与分别指定值
        :param label:
        :param version:
        :param flags:
        :param shapes:
        :param imagePath:
        :param imageData:
        :param imageHeight:
        :param imageWidth:
        '''
        self.output_shape = None
        self.prefix = prefix
        self.img_name = img_name
        self.img = cv2.imread(os.path.join(self.prefix, self.img_name))
        self.input_shape = self.img.shape
        self.imageWidth = float(self.input_shape[1])
        self.imageHeight = float(self.input_shape[0])
        self.imageData = imageData
        self.imagePath = img_name  #imagePath
        self.shapes = shapes
        self.flags = flags
        self.version = version

        # 辅助成员变量

    def init_by_imgpth(self, relative_pth: str):
        img = cv2.imread(relative_pth)
        h, w, c = img.shape
        self.imageWidth = w
        self.imageHeight = h

    def __get_final_label__(self):
        rst = {}
        rst["version"] = self.version
        rst['flags'] = self.flags
        rst['shapes'] = self.shapes
        rst['imagePath'] = self.imagePath
        rst['imageData'] = self.imageData
        rst['imageHeight'] = self.imageHeight
        rst['imageWidth'] = self.imageWidth
        return rst

    def labeling(self, annotations):
        '''_summary_
        将json_label中的点注入labelme中
        Args:
            points (list): _description_
        '''
        shapes = []

        for gid, text_roi_dict in enumerate(annotations):
            shape_dict = {}
            shape_dict['label'] = f'text'
            l = text_roi_dict['points']
            shape_dict['points'] = [
                l[i:i + 2] for i in range(0, len(l), 2)
            ]  #split_list(text_roi_dict['points'],n=2,newList=[])
            shape_dict["group_id"] = str(gid)  # 每一个实例都不同
            shape_dict['shape_type'] = 'polygon'
            shape_dict['flags'] = {}

            shapes.append(shape_dict)
        self.shapes = shapes

    def output(self, rst_dir: str = ''):
        '''
        输出此对象至rst_pth
        :param rst_pth: 输出路径，默认为原JSON路径
        :return:
        '''
        rst_pth = os.path.join(rst_dir, self.img_name.split('.')[0] + '.json')
        rst = {}
        rst["version"] = str(self.version)
        rst['flags'] = {}
        rst['shapes'] = self.shapes
        rst['imagePath'] = str(self.imagePath)
        rst['imageData'] = None  # self.imageData
        rst['imageHeight'] = round(float(self.imageHeight), 1)
        rst['imageWidth'] = round(float(self.imageWidth), 1)
        json.dumps(rst)
        with open(rst_pth,
                  'w+') as f:  # 打开文件用于读写，如果文件存在则打开文件，将原有内容删除；文件不存在则创建文件；
            # pass
            f.write(json.dumps(rst))
        return True


def json2labelme(img_dir, json_pth, rst_dir):
    '''_summary_
    jsonlabel格式
    {'image_name1':[{“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text},
{“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text},
					…],
    {'image_name2':[{“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text},
{“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text},
					…]
                    }
    Args:
        fpth (_type_): _description_
        rst_dir (_type_): _description_
    '''

    auto_make_directory(rst_dir)
    with open(json_pth, 'r') as f:
        json_labels = json.loads(f.read())  #dict格式
    shapes = []
    for item in tqdm(json_labels.items()):
        file_name = item[0]
        annotations = item[
            1]  # [{“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text}, {“points”:  x1, y1, x2, y2, …, xn, yn, “transcription”: text},…],
        labelme_obj = Label(prefix=img_dir, img_name=file_name)
        labelme_obj.labeling(annotations)
        labelme_obj.output(rst_dir)
    pass
def is_color(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    img = img.astype(np.uint64)
    b, g, r = cv2.split(img)

    if np.sum(b) == np.sum(g) == np.sum(r):
        # hist = cv2.calcHist([img],[0],None,[16],[0,256])
        # print(hist)
        return False
    return True
def is_bin_bg_white_local(img):
    '''_summary_
    判断二值图背景是否为白色
    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        pass
    h,w = img.shape[:2]
    img = img[h//4:h-h//4,w//4:w-w//4]
    # print('---------')
    showim(img)
    img = img.astype(np.uint64)

    n_h,n_w = img.shape[:2]

    max_val = n_h*n_w
    img = img.astype(np.uint64)
    img_01 = img/255
    current_val = np.sum(img_01)
    ratio = current_val/max_val
    print(ratio)
    if ratio > 0.5:
        return True
    # print('not')
    return False
def otsu_bin_(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = np.ones((3,3),np.uint8)
    # res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 20)
    res = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
    return res
    # return res
def otsu_bin(img: np.ndarray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res
def adaptative_thresholding(img: np.ndarray, threshold=20):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orignrows, origncols = gray.shape
    M = int(np.floor(orignrows/16) + 1)
    N = int(np.floor(origncols/16) + 1)
    Mextend = 0 if round(M/2)-1 < 0 else round(M/2)-1
    Nextend = 0 if round(N/2)-1 < 0 else round(N/2)-1
    # print(Mextend,Nextend)
    aux =cv2.copyMakeBorder(gray, top=Mextend, bottom=Mextend, left=Nextend,
                          right=Nextend, borderType=cv2.BORDER_REFLECT)
    windows = np.zeros((M,N),np.int32)
    imageIntegral = cv2.integral(aux, windows,-1)
    nrows, ncols = imageIntegral.shape

    temp = np.zeros((orignrows, origncols))
    for i in range(nrows-M):
        for j in range(ncols-N):
            temp[i, j] = imageIntegral[i+M, j+N] - imageIntegral[i, j+N]+ imageIntegral[i, j] - imageIntegral[i+M,j]
    rst = np.ones((orignrows, origncols), dtype=np.bool)
    graymult = (gray).astype('float64')*M*N
    rst[graymult <= temp*(100.0 - threshold)/100.0] = False
    rst = (255*rst).astype(np.uint8)

    return rst

def extract_image(img_dir, fpth):
    # fs = get_files_pth(img_dir)
    with open(fpth, 'r') as f:
        json_labels = json.loads(f.read())  #dict格式
    # text_labels = []
    # gid = 1
    # label_lst = list(json_labels.items())
    # print(label_lst[:10])
    for item in tqdm(json_labels.items()):
        file_name = item[0]
        # print(file_name)
        annotations = item[1]
        img_ori = cv2.imread(os.path.join(img_dir, file_name))
        if img_ori is None:
            continue
        # is_color(img_ori)
        for gid,text_roi_dict in enumerate(annotations):
            new_file_name = file_name.split('.')[0] + f'_{gid}.png'
            if os.path.exists(os.path.join(rst_dir,
                             new_file_name)):
                # print('------')
                break
            data_dict = {}


            img = img_ori.copy()
            l = text_roi_dict['points']
            data_dict['points'] = [
                l[i:i + 2] for i in range(0, len(l), 2)
            ]  #split_list(text_roi_dict['points'],n=2,newList=[])
            data_dict['transcription'] = text_roi_dict['transcription']

            # 定义四个顶点坐标
            pts = np.array(data_dict['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))

            x, y, w, h = cv2.boundingRect(pts)  #轮廓
            inner_pts = pts - np.array([x,y])
            # print(x,y,w,h)
            # bin_mask = np.zeros(img.shape[:2], np.uint8)

            img_patch = img[y:y + h, x:x + w]
            # print(x, y, w, h)

            mask = np.zeros(img.shape, np.uint8)[y:y + h, x:x + w]
            # mask_patch = mask
            # 画多边形
            # mask = cv2.polylines(mask, [pts], True, (0, 0, 255), 1)
            try:
                mask2 = cv2.drawContours(mask.copy(), [inner_pts], -1, (255,255,255), thickness=-1)

            except cv2.error:
                print(f'Warning! inner_pts ERROR when processing {file_name}')
                continue


            ones = 2*np.ones(img_patch.shape,dtype=np.uint8)
            img_patch = cv2.add(img_patch,ones)
            ROI = cv2.bitwise_and(mask2, img_patch)

            # showim(ROI)
            if len(ROI)==0:
                print(f'Warning! Negative points values founded when processing {file_name}')
                continue
            ROI1 = cv2.rotate(ROI, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if ROI1 is None:
                print(f'Warning! None Type Error ROI founded when processing {file_name}')
                continue
            # showim(ROI1)

            cv2.imwrite(
                os.path.join(rst_dir,
                             new_file_name), ROI1)
                        # print(new_file_name)
            write_csv([(new_file_name, data_dict['transcription'])],
                      r'K:\FINAL_GJJS\seq_label.csv',
                      False)


def extract_mask(img_dir, fpth):
    fs = get_files_pth(img_dir)
    with open(fpth, 'r') as f:
        json_labels = json.loads(f.read())  #dict格式
    text_labels = []
    # gid = 1
    # label_lst = list(json_labels.items())
    # print(label_lst[:10])
    for item in tqdm(json_labels.items()):
        file_name = item[0]
        # print(file_name)
        annotations = item[1]
        img_ori = cv2.imread(os.path.join(img_dir, file_name))
        if img_ori is None:
            continue
        # is_color(img_ori)
        for gid,text_roi_dict in enumerate(annotations):
            new_file_name = file_name.split('.')[0] + f'_{gid}.jpg'
            if os.path.exists(os.path.join(rst_dir,
                             new_file_name)):
                # print('------')
                break
            data_dict = {}


            img = img_ori.copy()
            l = text_roi_dict['points']
            data_dict['points'] = [
                l[i:i + 2] for i in range(0, len(l), 2)
            ]  #split_list(text_roi_dict['points'],n=2,newList=[])
            data_dict['transcription'] = text_roi_dict['transcription']

            # 定义四个顶点坐标
            pts = np.array(data_dict['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(pts)  #轮廓


            mask = np.zeros(img.shape, np.uint8)

            # 画多边形
            # mask = cv2.polylines(mask, [pts], True, (0, 0, 255), 1)
            mask2 = cv2.fillConvexPoly(mask.copy(), [pts],
                                 (255, 255, 255))  # 用于求 ROI



            saved_mask = mask2[y:y + h, x:x + w]
            saved_mask = cv2.rotate(saved_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            saved_mask = cv2.cvtColor(saved_mask, cv2.COLOR_BGR2GRAY)
            _, saved_mask = cv2.threshold(saved_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ver_file_name = file_name.split('.')[0] + f'_{gid}.png'
            cv2.imwrite(
                os.path.join(r'K:\Data\image_seq_mask',
                             ver_file_name), saved_mask)





def json2txt(fpth, rst_dir):
    pass


def build_coco():


    import labelme2coco
    save_json_path = os.path.join(r'-',
                                  "annotations-origin.json")

    # convert labelme annotations to coco
    labelme2coco.convert(
        r'K:\FINAL_GJJS\labelme_annotation',
        save_json_path)
    print('转换成功，将JSON保存至：' + r'' + save_json_path)

build_coco()
def build_lmdb():

    pass
def get_filename_from_pth(file_pth: str, suffix: bool = True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    if suffix: #如果保留后缀

        rst = '.'.join(fname_list)
    else:#如果不保留后缀
        rst = '.'.join(fname_list[:-1])
    # print(rst)
    return rst


def dir2bin(ori_dir,rst_dir):
    auto_make_directory(rst_dir)
    for pth in tqdm(get_files_pth(ori_dir)):
        img_name = get_filename_from_pth(pth)
        print(img_name)

        rst_pth = os.path.join(rst_dir,img_name)
        if os.path.exists(rst_pth):
            continue
        img = cv2.imread(pth)

        if is_color(img):#是彩色图 什么也不做
            os.system(f'copy {pth} {rst_pth}')
            # cv2.imwrite(rst_pth,img)
        else:#不是彩色图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = img.shape[:2]

            bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
            img = 255-img
            # showim(img)
            if not is_bin_bg_white_local(bin):
                img = 255 - img


            # showim(img)
            cv2.imwrite(rst_pth,img)
    pass
