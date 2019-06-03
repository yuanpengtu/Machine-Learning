# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:00:30 2018

@author: Amitesh863
"""
import numpy as np
from mean_shift import mean_shift
from kmeans_new import kmeans
from fcm import fuzzy_cmeans



def mse(imageA, imageB):

	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
'''
input_dir = '../result/'

kmeans=gi(input_dir+"kmeans.png")
cmeans=gi(input_dir+"cmeans.png")
mean_shift=gi(input_dir+"ms.png")

print("Mean Square Error between image obtained by kmeans and cmeans is {}".format(mse(kmeans,cmeans)))
print("Mean Square Error between image obtained by kmeans and mean shift is {}".format(mse(kmeans,mean_shift)))
print("Mean Square Error between image obtained by mean shift and cmeans is {}".format(mse(cmeans,mean_shift)))
'''


#Meanshift Profiling

fnames = ['Scenery.jpg','input2.jpg','input5.jpg']
radi   =[10,20,30,40]

results =[]
for fname in fnames:
    for rad in radi:
        time,db_index = mean_shift(fname,rad,get_stat = True)
        print(time,db_index)


#print(fuzzy_cmeans('cameraman.tif',2,get_stat=True))
