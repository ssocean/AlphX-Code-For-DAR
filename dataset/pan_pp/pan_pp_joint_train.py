import math

from operator import gt
import random
import PIL.ImageOps 
import string
from imgaug import augmenters as iaa
import cv2
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import scipy.io as scio
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from torch.utils import data
# from coco_text import COCO_Text
from .coco_text import COCO_Text

EPS = 1e-6


'''
组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里组委会看这里
'''

ct_train_data_dir = 'data/TOTAL/' #原始图像放这里
ct_train_gt_path = 'data/TOTAL-CT.json' #label-CT.json路径



def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    elif voc_type == 'CHINESE':
        with open('codelist.txt', 'r',encoding='utf-8') as f:  # 打开文件
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

voc, char2id, id2char = get_vocabulary('CHINESE')

def to_words(seqs):
    EPS = 1e-6
    words = []


    # print(seqs.shape)
    
    word = ''
    word_score = 0
    for j, char_id in enumerate(seqs):
        char_id = int(char_id)
        # print(f'char_id:{char_id}')
        # print(f'self.END_TOKEN:{self.END_TOKEN}')
        if char_id == char2id['EOS'] : 
            word+='||'
        elif id2char[char_id] in ['PAD']:
            word += '-'
        elif id2char[char_id] in ['UNK']:
            word += '#'
        else:
            word += id2char[char_id]
        # print(current_char)
        
    words.append(word)

    return words

def get_img(img_path, read_type='cv2'):
    read_type='cv2'
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print(img_path)
        raise
    return img


def check(s):
    # for c in s:
    #     if c in list(string.printable[:-6]):
    #         continue
    #     return False
    return True


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
    polygons = []
    # print(anns)
    is_ndarray = True
    for ann in anns:
        bbox = ann['polygon'] #polygon bbox
        # print(bbox)
        if(len(bbox)!=8): # 大于4个点
            # print('---------------------------')
            is_ndarray = False
        #     n = []
        #     for i in range(0,len(bbox),2):
        #         n.append([bbox[i],bbox[i+1]])
        #     p1 = min(n)

        #     p2 = max(n) 
        #     p3 = [p1[0],p2[1]]
            
        #     p4 = [p1[1],p2[0]]
        #     bbox = p1 + p3 + p2 +p4
        # print(bbox)
        polygon = ann['polygon']    
        polygon = np.array(polygon) / ([w * 1.0, h * 1.0] * (len(polygon) // 2)) #归一化
        polygons.append(polygon)
        # print(bbox)
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * (len(bbox) // 2)) #归一化
        # print(bbox)
        bboxes.append(bbox)

        if 'utf8_string' not in ann:
            words.append('###')
        else:
            word = ann['utf8_string']
            if not check(word):
                words.append('???')
            else:
                words.append(word)
    # print(np.array(bboxes).shape)
    # print(len(bboxes))
    # bboxes = np.array(bboxes) if is_ndarray else bboxes
    # print(type(bboxes))
    if(is_ndarray):
        return np.array(bboxes), words, polygons
    else:
        return bboxes, words, polygons




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
    try:
        img = cv2.resize(img, dsize=(w, h))
    except cv2.error:
        # print(img)
        raise
    return img


def random_scale(img, min_size, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))#, 1.3
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05]))#, 1.1
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    # print (h_scale, w_scale, h_scale / w_scale)

    img = scale_aligned(img, h_scale, w_scale)
    # img = scale_aligned(img, 1, 1)
    return img


def random_crop_padding(imgs, target_size):
    """using padding and the final crop size is (800, 800)"""
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs
    
    
    n_imgs = []
    for idx in range(len(imgs)):
        img_p = cv2.resize(imgs[idx],target_size)
        n_imgs.append(img_p)
        
        
        
    # t_h = t_h if t_h < h else h
    # t_w = t_w if t_w < w else w

    # if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
    #     # make sure to crop the text region
    #     tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
    #     tl[tl < 0] = 0
    #     br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
    #     br[br < 0] = 0
    #     br[0] = min(br[0], h - t_h)
    #     br[1] = min(br[1], w - t_w)

    #     i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
    #     j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    # else:
    #     i = random.randint(0, h - t_h) if h - t_h > 0 else 0
    #     j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    # n_imgs = []
    # for idx in range(len(imgs)):
    #     if len(imgs[idx].shape) == 3:
    #         s3_length = int(imgs[idx].shape[-1])
    #         img = imgs[idx][i:i + t_h, j:j + t_w, :]
    #         img_p = cv2.copyMakeBorder(img,
    #                                    0,
    #                                    p_h - t_h,
    #                                    0,
    #                                    p_w - t_w,
    #                                    borderType=cv2.BORDER_CONSTANT,
    #                                    value=tuple(0
    #                                                for i in range(s3_length)))
    #     else:
    #         img = imgs[idx][i:i + t_h, j:j + t_w]
    #         img_p = cv2.copyMakeBorder(img,
    #                                    0,
    #                                    p_h - t_h,
    #                                    0,
    #                                    p_w - t_w,
    #                                    borderType=cv2.BORDER_CONSTANT,
    #                                    value=(0,))
    #     n_imgs.append(img_p)
        
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


class PAN_PP_Joint_Train(data.Dataset):
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
        self.img_paths['ct'] = self.ct.getImgIds( ) #imgIds=self.ct.train,catIds=[('legibility','legible')]
        # print(self.img_paths)
        self.img_num += len(self.img_paths['ct'])
        


        self.voc, self.char2id, self.id2char = get_vocabulary('CHINESE')
        self.max_word_num = 400
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)
        print(f'self.img_num={len(self.img_paths["ct"])}')

    def __len__(self):
        return self.img_num



    def load_ct_single(self, index):
        
        img_meta = self.ct.loadImgs(self.img_paths['ct'][index])[0]
        img_path = ct_train_data_dir + img_meta['file_name']
        # print(index)
        # print(f'img_pth:{img_path}')
        img = get_img(img_path, self.read_type)

        annIds = self.ct.getAnnIds(imgIds=img_meta['id'])
        anns = self.ct.loadAnns(annIds)
        # print([ann['utf8_string'] for ann in anns])
        bboxes, words,polygons = get_ann_ct(img, anns)
        # print(bboxes)
        return img, bboxes, words,polygons



    def __getitem__(self, index):
        # index = random.randint(0,15)
        # index = random.choice(['剪刀', '石头', '布'])
        # index = 8
        # print(f'Current index:{index}')

        # index = 1772
        # print(f'Current index:{index}')

        img, bboxes, words, polygons = self.load_ct_single(index)

        # print(type(bboxes))
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]
            polygons = polygons[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len),
                           self.char2id['PAD'],
                           dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1,), dtype=np.int32)
        # print(f'get_item_words:{words}')
        for i, word in enumerate(words):
            # print(word)
            if word == '###':
                continue
            if word == '???':
                continue
            # word = word.lower()
            # print(word)
            gt_word = np.full((self.max_word_len,),
                              self.char2id['PAD'],
                              dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            # print(gt_word)
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1
        try:
            if self.is_transform:
                img = random_scale(img, self.img_size[0], self.short_size)
        except:
            print(index)
            print(img.dtype)
            print(img)
            raise
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            if type(bboxes) == list:
                for i in range(len(bboxes)):
                    bboxes[i] = np.reshape(
                        bboxes[i] * ([img.shape[1], img.shape[0]] *
                                     (bboxes[i].shape[0] // 2)),
                        (bboxes[i].shape[0] // 2, 2)).astype('int32')
            else:
                # print(img.shape[1])
                # print(img.shape[0])
                # print(f'bshape{bboxes.shape}')

                # print(([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)))
                # print(bboxes.shape[1])
                
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)),(bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                # ret = cv2.imwrite(f'temp/{i}.png',gt_instance)
                # assert ret, 'failed'
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.for_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = \
                imgs[0], imgs[1], imgs[2], imgs[3:]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop,
                                         word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])
        
        if self.is_transform:
            img = Image.fromarray(img)
            # img = img.convert('L')#RGB
            img = img.convert('RGB')
            # img = torchvision.transforms.RandomInvert()
            # img = transforms.ColorJitter(brightness=32.0 / 255,
            #                              saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('L')#RGB
            img = img.convert('RGB')
        # img = np.array([img], dtype=np.uint8)
        # img = transforms.ToTensor()(img)
        # img = torch.Tensor(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                            std=[0.229, 0.224, 0.225])(img)
        # img = np.array([np.array(img, dtype=np.uint8)])
        # img = torch.Tensor(img)
        # print(np.array(img).shape) # 896 896 3
        # img = torch.Tensor(np.array(img, dtype=np.uint8))
        # img = img.transpose(0,2).transpose(1,2)
        often = lambda aug: iaa.Sometimes(0.8, aug)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seldom = lambda aug: iaa.Sometimes(0.2, aug)
        seq_1104 = iaa.Sequential([
        seldom(iaa.OneOf([
        iaa.Invert(0.1),
        iaa.CoarsePepper(0.005, size_percent=(0, 0.005)),
        iaa.CoarseSaltAndPepper(0.005, size_percent=(0, 0.005)),
        ]),),
        often(iaa.OneOf([
                    iaa.MultiplyBrightness((0.8, 1.1)),
                    iaa.LinearContrast((0.9, 1.1)),
                    iaa.Multiply((0.8, 1.1), per_channel=0.2),
                        ])),
        sometimes(iaa.OneOf([
            iaa.JpegCompression(compression=(0, 50)),
            iaa.imgcorruptlike.GaussianNoise(severity=1),
            iaa.imgcorruptlike.ShotNoise(severity=1),
            iaa.imgcorruptlike.ImpulseNoise(severity=1),
            iaa.imgcorruptlike.SpeckleNoise(severity=1),
        ])),
        ], random_order=False) 
        # BCHW->BHWC
        img = np.array(img)
        img = seq_1104(image=img)
        
        
        # if random.randint(0,9)>4:
        #     img = PIL.ImageOps.invert(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.755, 0.731, 0.694],
                            std=[0.267, 0.260, 0.247])(img) # 3 896 896
        # print(img.shape)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()
        
        # img = transforms.ToTensor()(img)
        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )
        # print(data['imgs'].shape)
        if self.for_rec:
            data.update(dict(gt_words=gt_words, word_masks=word_mask))

        return data


if __name__ == '__main__':
    data_loader = PAN_PP_Joint_Train(split='train',
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
        for i, (k, v) in enumerate(item.items()):
            # print(f'k: {k}, v.shape: {v.shape}')
            print(i)
