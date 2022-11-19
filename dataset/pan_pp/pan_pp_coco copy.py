import glob
import math
import os
import random
import string

import cv2
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import scipy.io as scio
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

from .coco_text import COCO_Text

EPS = 1e-6


ct_root_dir = '/opt/data/private/pan_pp.pytorch/data/'
ct_train_data_dir = '/opt/data/private/pan_pp.pytorch/data/test/'#test/ image
ct_train_gt_path = '/opt/data/private/pan_pp.pytorch/data/label-CT.json'


def get_img(img_path, read_type='cv2'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = Image.open(img_path)
            img = img.resize((896, 896),Image.ANTIALIAS)
            img = np.array(img)
    except Exception:
        print(img_path)
        raise
    return img


def get_ann_synth(img, gts, texts, index):
    bboxes = np.array(gts[index])
    bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
    bboxes = bboxes.transpose(2, 1, 0)
    bboxes = np.reshape(
        bboxes, (bboxes.shape[0], -1)) / ([img.shape[1], img.shape[0]] * 4)

    words = []
    for text in texts[index]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        words.extend([w for w in text.split(' ') if len(w) > 0])

    return bboxes, words




def get_ann_ct(img, anns):
    h, w = img.shape[0:2]
    bboxes = []
    words = []
    for ann in anns:
        bbox = ann['polygon']
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * (len(bbox) // 2))
        bboxes.append(bbox)

        if 'utf8_string' not in ann:
            words.append('###')
        else:
            word = ann['utf8_string']
            if not check(word):
                words.append('???')
            else:
                words.append(word)

    return np.array(bboxes), words



def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img,
                                      rotation_matrix, (h, w),
                                      flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    # print (h_scale, w_scale, h_scale / w_scale)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding(imgs, target_size):
    """using padding and the final crop size is (800, 800)"""
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0
                                                   for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5),
                         max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    elif voc_type == 'CHINESE':
        # high_fre.txt
        # chars.txt
        with open('/opt/data/private/pan_pp.pytorch/chars.txt', 'r',encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件
            voc = [i for i in data]
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", '
                       '"ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def get_files_pth(dir_pth: str, suffix: str = '*'):
    '''
    返回dir_pth下以后缀名suffix结尾的文件绝对路径list
    :param dir_pth:文件夹路径
    :param suffix:限定的文件后缀
    :return: 文件绝对路径list
    '''
    rst = []
    glob_pth = os.path.join(dir_pth, f'*.{suffix}')
    for filename in glob.glob(glob_pth):
        rst.append(filename)
    return rst
class PAN_PP_COCO_BK(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False,
                 debug=False):
        self.split = split
        self.is_transform = is_transform and not debug

        self.img_size = img_size if (
                img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                       img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.for_rec = with_rec
        self.read_type = read_type
        self.report_speed = report_speed
        self.debug = debug

        self.img_paths = {}
        self.gts = {}
        self.texts = {}

        self.img_num = 0
        
        
        # coco_text
        self.ct = COCO_Text(ct_train_gt_path)
        self.img_paths= get_files_pth(ct_train_data_dir)
        print(self.img_paths)
        self.img_num += len(self.img_paths)
        
        
        self.voc, self.char2id, self.id2char = get_vocabulary('CHINESE')
        self.max_word_num = 400
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)


    def __len__(self):
        return self.img_num

    # def load_synth_single(self, index):
    #     img_path = synth_train_data_dir + self.img_paths['synth'][index][0]
    #     img = get_img(img_path, self.read_type)
    #     bboxes, words = get_ann_synth(img, self.gts['synth'],
    #                                   self.texts['synth'], index)
    #     return img, bboxes, words

    # def load_ic17_single(self, index):
    #     img_path = self.img_paths['ic17'][index]
    #     gt_path = self.gts['ic17'][index]
    #     img = get_img(img_path, self.read_type)
    #     bboxes, words = get_ann_ic17(img, gt_path)
    #     return img, bboxes, words

    def load_ct_single(self, index):
        img_meta = self.ct.loadImgs(self.img_paths['ct'][index])[0]
        img_path = ct_train_data_dir + img_meta['file_name']
        img = get_img(img_path, self.read_type)

        annIds = self.ct.getAnnIds(imgIds=img_meta['id'])
        anns = self.ct.loadAnns(annIds)
        bboxes, words = get_ann_ct(img, anns)

        return img, bboxes, words

    # def load_ic15_single(self, index):
    #     img_path = self.img_paths['ic15'][index]
    #     gt_path = self.gts['ic15'][index]
    #     img = get_img(img_path, self.read_type)
    #     bboxes, words = get_ann_ic15(img, gt_path)
    #     return img, bboxes, words

    # def load_tt_single(self, index):
    #     img_path = self.img_paths['tt'][index]
    #     gt_path = self.gts['tt'][index]
    #     img = get_img(img_path, self.read_type)
    #     bboxes, words = get_ann_tt(img, gt_path)
    #     return img, bboxes, words
    # def prepare_test_data(self, index):
        
    def __getitem__(self, index):
        
        # print(f'Current index:{index}')
        img_path = self.img_paths[index]

        if self.debug:
            img_path = './data/ICDAR2015/Challenge4/ch4_test_images/img_499.jpg'
        # print('\n')
        # print('\n')
        # print('\n')
        # print('\n')
        # print('*'*55)
        # print(img_path)
        img_meta = dict(
            img_path=img_path,
            img_name=img_path.split('/')[-1].split('.')[0])

        img = get_img(img_path, self.read_type)
        img_meta.update(dict(org_img_size=np.array(img.shape[:2])))
        # scale_aligned_short(img, self.short_size)
        # img = random_scale(img)
        
        # img = cv2.resize(img, dsize=(736, 736))
        img_meta.update(dict(img_size=np.array(img.shape[:2])))
        
        img = Image.fromarray(img)
        # img = transforms.Resize((736,992))(img)
        img = img.convert('RGB')
        
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        # print(img.shape)
        

        if self.debug:
            print(img.sum())
            exit()

        data = dict(imgs=img, img_metas=img_meta)

        return data


if __name__ == '__main__':
    data_loader = PAN_PP_COCO(split='train',
                                     is_transform=True,
                                     img_size=736,
                                     short_size=736,
                                     kernel_scale=0.5,
                                     read_type='pil',
                                     with_rec=False)
    train_loader = torch.utils.data.DataLoader(data_loader,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               drop_last=True,
                                               pin_memory=True)
    for item in train_loader:
        print('-' * 20)
        for k, v in item.items():
            print(f'k: {k}, v.shape: {v.shape}')
