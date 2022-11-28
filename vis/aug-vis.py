import cv2 
import numpy as np
from imgaug import augmenters as iaa
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