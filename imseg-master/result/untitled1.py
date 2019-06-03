# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:54:58 2018

@author: Amitesh863
"""

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from PIL import Image
import numpy as np


im2=Image.open('ks_Scenery.jpg').convert('L')
imageA=np.asarray(im2)


im3=Image.open('ms_Scenery.jpg').convert('L')
imageB=np.asarray(im3)


im3=Image.open('ms_Scenery.jpg').convert('L')
imageB=np.asarray(im3)

im = Image.open('fcm_scenery.jpeg').convert('L')
out = im.resize((256, 256),Image.ANTIALIAS)   
imageC = np.asarray(out)

ssim_const = ssim(imageA, imageB)
ssim_const1 = ssim(imageA, imageC)
ssim_const2 = ssim(imageB, imageC)
print(ssim_const,ssim_const1,ssim_const2)