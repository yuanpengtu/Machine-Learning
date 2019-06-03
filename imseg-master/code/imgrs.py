# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:57:39 2018

@author: Amitesh863
"""
from PIL import Image
import numpy as np


def get_image(img):
   
    im = Image.open(img).convert('L')
    out = im.resize((256, 256),Image.ANTIALIAS)   
    arr_img = np.asarray(out)
    return arr_img