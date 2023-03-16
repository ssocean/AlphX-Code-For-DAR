import json
import os
from os.path import splitext
from os import listdir
import numpy as np
import glob
import torch
from imgaug import SegmentationMapsOnImage
from torch.utils.data import Dataset
import cv2
import imgaug.augmenters as iaa
from torchvision import transforms

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
mask_size = (256, 256)


def find_cnt_center(cnt):
    '''_summary_
    计算轮廓cnt的中心坐标
    Args:
        cnt (_type_): _description_

    Returns:
        _type_: _description_
    '''
    M = cv2.moments(cnt)  # 计算矩特征
    if M["m00"] == 0:
        return (-1, -1)
    if len(cnt) <= 6:
        return (-1, -1)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


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


def get_filename_from_pth(file_pth: str, suffix: bool = True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    if suffix:  # 如果保留后缀

        rst = '.'.join(fname_list)
    else:  # 如果不保留后缀
        rst = '.'.join(fname_list[:-1])
    return rst


def showim(img: np.ndarray, img_name: str = 'image', is_fixed=True):
    '''
    展示图片
    :param img: ndarray格式的图片
    '''
    if is_fixed:
        cv2.namedWindow(img_name, cv2.WINDOW_AUTOSIZE)
    else:
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class BinDataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, imgs_dir, masks_dir):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # 获取图片名称，ids是一个列表
                    if not file.startswith('.')]
        pass

    @classmethod
    def _otsu_bin(cls, mask):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, res = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res

    @classmethod
    def _preprocess(cls, img, mask):
        """
        用于在加载数据集的时候对图像做预处理，不可被外部调用
        :param img:输入的图像
        :return:经预处理的图像
        """
        seq = iaa.OneOf([
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent=(0, 0.1), rotate=(-40, 40), cval=0,
                       mode='constant'),  # 仿射变换
            iaa.ShearX((-20, 20)),
            iaa.CropAndPad(px=(-10, 0), percent=None, pad_mode='constant', pad_cval=0, keep_size=True),  # 裁剪缩放
            iaa.PiecewiseAffine(scale=(0, 0.05), nb_rows=4, nb_cols=4, cval=0),  # 以控制点的方式随机形变
        ])
        img_aug, msk_aug = seq(image=img, segmentation_map=mask)

        return img_aug, msk_aug

    @classmethod
    def _channel_combination(cls, img):
        '''
        用于图像色彩通道转换与融合
        :param img: RGB通道或其它通道图像
        :return: 经通道转换后的图像
        '''
        rst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # L, a, b = cv2.split(lab)
        # rst = cv2.merge([L, a, b])
        return rst

    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return len(self.ids)

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''
        # item = 26
        idx = self.ids[item]
        mask_file = glob(self.masks_dir + idx + '.*')  # 获取指定文件夹下文件名(列表)
        img_file = glob(self.imgs_dir + idx + '.*')

        img_path = img_file[0]
        mask_path = mask_file[0]

        assert len(img_file) == 1, \
            f'未找到图片 {idx}: {img_file}'

        assert len(mask_file) == 1, \
            f'未找到图片掩膜{idx}: {mask_file}'
        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        assert img.shape[:2] == mask.shape[:2], \
            f'图片与掩膜 {idx} 大小不一致,图片： {img.shape[:2]} 掩膜： {mask.shape[:2]}'

        _, M = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # M=M/255
        # mask = self._otsu_bin(mask)  # 数据问题 需要先做一次二值化
        mask = mask / 255


        img = cv2.bitwise_and(img, M.astype(np.uint8))
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'M': torch.from_numpy(M).type(torch.FloatTensor)
        }
        pass


def load_cnts_from_json(json_pth):
    with open(json_pth, encoding='utf-8') as f:
        result = json.load(f)
    cnts = []
    for region in result['shapes']:
        points = region['points']
        poly = np.array(points).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cnts.append(poly)
    return cnts


def resize_contour(cnts, ori_size, rst_shape):
    '''
    原地操作函数，由于原图尺寸的变换将会导致标注信息的变换，该方法完成在图片尺寸变换时标注信息的同步转换。
    最好由低分辨率放大至高分辨率
    :return:
    '''
    o_h, o_w = ori_size
    r_h, r_w = rst_shape
    height_ratio = r_h / o_h
    width_ratio = r_w / o_w  # 计算出高度、宽度的放缩比例
    ratio_mat = [[width_ratio, 0], [0, height_ratio]]
    # print(points_to_poly(cnts).shape)
    return (np.array(cnts).astype(np.int32).reshape((-1)).reshape((-1, 2)) @ ratio_mat).astype(np.int32)  # n×2 矩阵乘 2×2


def preprocess(img, mask):
    """
    用于在加载数据集的时候对图像做预处理，不可被外部调用
    :param img:输入的图像
    :return:经预处理的图像
    """
    mask = mask.astype(np.uint16)
    segmap = SegmentationMapsOnImage(mask, shape=img.shape)
    seq = iaa.OneOf([
        iaa.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, translate_percent=(0, 0.2), rotate=(-25, 25), cval=0,
                   mode='constant'),  # 仿射变换
        iaa.ShearX((-15, 15)),
        iaa.CropAndPad(percent=(-0.2, 0.5), pad_mode='constant', pad_cval=0, keep_size=True),  # 裁剪缩放
        iaa.PiecewiseAffine(scale=(0, 0.05), nb_rows=4, nb_cols=4, cval=0),  # 以控制点的方式随机形变
    ])
    img_aug, msk_aug = seq(image=np.array(img, dtype=np.uint8), segmentation_maps=segmap)
    return img_aug, msk_aug.get_arr()


class OrderDataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, base_dir):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.base_dir = base_dir
        self.imgs_pth = get_files_pth(base_dir, 'jpg') + get_files_pth(base_dir, 'png')
        self.jsons_pth = get_files_pth(base_dir, 'json')
        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # 获取图片名称，ids是一个列表
        #             if not file.startswith('.')]
        assert len(self.imgs_pth) == len(self.jsons_pth), 'Num not the same'

    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return len(self.imgs_pth)

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''
        # item = 26
        img_pth = self.imgs_pth[item]
        # json_pth = self.jsons_pth[item]
        img_name = get_filename_from_pth(img_pth, False)
        # json_name = get_filename_from_pth(json_pth, False)
        json_pth = os.path.join(self.base_dir, img_name + '.json')
        # assert img_pth[:-4] == json_pth[:-4],'PTH Error'
        polys = load_cnts_from_json(json_pth)
        img = cv2.imread(img_pth, 0)
        filename_wo_suffix = get_filename_from_pth(img_pth, suffix=False)

        mask = np.zeros(mask_size, dtype=np.float32)

        for i, poly in enumerate(polys):
            poly = resize_contour(poly, img.shape[:2], mask.shape[:2])
            mask = cv2.drawContours(mask, [poly.reshape((-1, 2))], -1, i + 2, thickness=-1)
        img = cv2.resize(img, mask_size)

        _, M = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        # M, mask = preprocess(M, mask.astype(np.uint16))

        M = M / 255
        max_val = len(polys) + 1
        mask = mask / max_val

        img = M.astype(np.uint8)
        mask = mask.astype(np.float32)


        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'M': torch.from_numpy(M.astype(np.uint8)).type(torch.FloatTensor)
        }
        pass


def filter_inward_cnt_by_centers(cnt_centers, region_cnt):
    rst = []
    for cc in cnt_centers:
        cx, cy = cc[1], cc[2]
        flag = cv2.pointPolygonTest(region_cnt, (cx, cy), False)
        # print(flag)
        if flag >= 0:
            # print()
            rst.append(cc)
    return rst


def merge_regions(ordered_mask, ori_size: type):
    '''

    :param ordered_mask: 具有不同值的mask
    :param ori_size: 原始图像shape
    :return:
    '''
    _, binary_mask = cv2.threshold(ordered_mask, 1, 65535, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rst = np.zeros(mask_size, dtype=np.uint8)
    for i in range(len(contours)):
        blank = np.zeros(mask_size, dtype=np.uint8)
        cv2.drawContours(blank, contours, i, 255, -1)
        # showim(blank,'blank')
        mean_val = cv2.mean(ordered_mask, blank)
        cv2.drawContours(rst, contours, i, mean_val, -1)

    return rst


class LayoutDataset(Dataset):
    '''
    数据集加载类
    '''

    def __init__(self, base_dir):
        """
        在此完成数据集的读取
        :param imgs_dir: 图片路径,末尾需要带斜杠
        :param masks_dir: mask路径，末尾需要带斜杠
        """
        self.base_dir = base_dir
        self.imgs_pth = get_files_pth(base_dir, 'jpg') + get_files_pth(base_dir, 'png')
        self.jsons_pth = get_files_pth(base_dir, 'json')
        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # 获取图片名称，ids是一个列表
        #             if not file.startswith('.')]
        assert len(self.imgs_pth) == len(self.jsons_pth), 'Num not the same'

    def __len__(self):
        '''
        返回数据集中包含的样本个数
        :return: 数据集中包含的样本个数
        '''
        return len(self.imgs_pth)

    def __getitem__(self, item):
        '''
        根据item，返回图片和它对应的标注图片
        :param item: 框架指定，请勿修改
        :return: 字典{'img':FloatTensor类型,'mask'：FloatTensor类型}
        '''
        # item = 26
        img_pth = self.imgs_pth[item]
        # json_pth = self.jsons_pth[item]
        img_name = get_filename_from_pth(img_pth, False)
        # json_name = get_filename_from_pth(json_pth, False)
        json_pth = os.path.join(self.base_dir, img_name + '.json')

        polys = load_cnts_from_json(json_pth)
        img = cv2.imread(img_pth, 0)

        mask = np.zeros(mask_size, dtype=np.uint16)

        for i, poly in enumerate(polys):
            blank = np.zeros(mask_size, dtype=np.uint8)
            poly = resize_contour(poly, img.shape[:2], mask.shape[:2])
            dialated_mask = cv2.drawContours(blank.copy(), [poly.reshape((-1, 2))], -1, 255, thickness=-1)  # i+1
            kernel = np.ones((4, 4), np.uint8)
            dialated_mask = cv2.dilate(dialated_mask.copy(), kernel, iterations=2)
            # showim(dialated_mask)
            contours, hierarchy = cv2.findContours(dialated_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = cv2.drawContours(mask.copy(), contours, -1, i + 1, thickness=-1)

        merged_regions = merge_regions(mask, (img.shape[0], img.shape[1]))

        img = cv2.resize(img, mask_size)

        _, M = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # M=M/255
        # mask = self._otsu_bin(mask)  # 数据问题 需要先做一次二值化
        max_val = merged_regions.max()
        mask = merged_regions / max_val

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'M': torch.from_numpy(M.astype(np.uint8)).type(torch.FloatTensor)
        }

    def load_cnts_from_json(json_pth):
        '''
        读取指定路径对应的[轮廓]
        :return: [poly1,poly2,...,poly3]
        '''
        with open(json_pth, encoding='utf-8') as f:
            result = json.load(f)
        cnts = []
        for region in result['shapes']:
            points = region['points']
            poly = np.array(points).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cnts.append(poly)
        return cnts

    def resize_contour(cnts, ori_size, rst_shape):
        '''
        原地操作函数，由于原图尺寸的变换将会导致标注信息的变换，该方法完成在图片尺寸变换时标注信息的同步转换。
        :return:
        '''
        o_h, o_w = ori_size
        r_h, r_w = rst_shape
        height_ratio = r_h / o_h
        width_ratio = r_w / o_w  # 计算出高度、宽度的放缩比例
        ratio_mat = [[width_ratio, 0], [0, height_ratio]]
        # print(points_to_poly(cnts).shape)
        return (np.array(cnts).astype(np.int32).reshape((-1)).reshape((-1, 2)) @ ratio_mat).astype(
            np.int32)  # n×2 矩阵乘 2×2

    def preprocess(img, mask):
        """
        用于在加载数据集的时候对图像做预处理，不可被外部调用
        :param img:输入的图像
        :return:经预处理的图像
        """
        mask = mask.astype(np.uint16)
        segmap = SegmentationMapsOnImage(mask, shape=img.shape)
        seq = iaa.OneOf([
            iaa.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, translate_percent=(0, 0.2), rotate=(-25, 25), cval=0,
                       mode='constant'),  # 仿射变换
            iaa.ShearX((-15, 15)),
            iaa.CropAndPad(percent=(-0.2, 0.5), pad_mode='constant', pad_cval=0, keep_size=True),  # 裁剪缩放
            iaa.PiecewiseAffine(scale=(0, 0.05), nb_rows=4, nb_cols=4, cval=0),  # 以控制点的方式随机形变
        ])
        img_aug, msk_aug = seq(image=np.array(img, dtype=np.uint8), segmentation_maps=segmap)
        return img_aug, msk_aug.get_arr()
    class OrderDataset(Dataset):
        '''
        数据集加载类

        '''
        def __init__(self, base_dir):
            '''

            :param base_dir:存放着图像+labelme格式标注的文件夹路径
            '''
            self.base_dir = base_dir
            self.imgs_pth = get_files_pth(base_dir, 'jpg') + get_files_pth(base_dir, 'png')
            self.jsons_pth = get_files_pth(base_dir, 'json')
            assert len(self.imgs_pth) == len(self.jsons_pth), 'Num not the same'

        def __len__(self):
            '''
            返回数据集中包含的样本个数
            :return: 数据集中包含的样本个数
            '''
            return len(self.imgs_pth)

        def __getitem__(self, item):
            '''
            根据item，返回图片和它对应的标注图片
            :param item: 框架指定，请勿修改
            :return: 字典{'img':FloatTensor类型，输入图像,'mask'：FloatTensor类型，对应不同实数值的GT,'M'：mask的阈值操作结果}
            '''

            img_pth = self.imgs_pth[item]

            img_name = get_filename_from_pth(img_pth, False)

            json_pth = os.path.join(self.base_dir, img_name + '.json')

            polys = load_cnts_from_json(json_pth)
            img = cv2.imread(img_pth, 0)
            # 注意数据类型，否则uint8最大只能绘制255个
            mask = np.zeros(mask_size, dtype=np.float32)
            # 绘制GT(mask)
            for i, poly in enumerate(polys):
                poly = resize_contour(poly, img.shape[:2], mask.shape[:2])
                # i+2便于threshold阈值操作，大于1全部转为前景像素
                mask = cv2.drawContours(mask, [poly.reshape((-1, 2))], -1, i + 2, thickness=-1)

            '''
            我们最开始考虑像素值越小，区域越优先。而背景是0，背景最优先。这不合乎逻辑，
            因此我们起初尝试对网络输出pred做掩码操作，即pred = pred * M，将掩码pred与真实标签mask做L1smooth损失。
            但实际操作中效果不佳，还是直接L1效果稍微好点。
            '''
            _, M = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

            # M归一化
            M = M / 255

            # mask归一化
            max_val = len(polys) + 1
            mask = mask / max_val

            img = M.astype(np.uint8)
            mask = mask.astype(np.float32)

            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'M': torch.from_numpy(M.astype(np.uint8)).type(torch.FloatTensor)
            }
            pass
