# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:52:05 2018

@author: durgesh singh
"""

import numpy as np
from matplotlib import pyplot as plt
from imgrs import get_image 
import random
from PIL import Image
import time


input_dir = '../data/'
output_dir = '../result/'

def show_results_fuzzy_cmeans(orig,seg,K,fname):
    plt.figure()
    ax1 =plt.subplot(121)
    ax2 =plt.subplot(122)
    ax1.set_title('Original Image')
    ax1.imshow(orig)
    ax2.set_title('fuzzy cmeans segmented image \nwith K = {}'.format(K))
    ax2.imshow(seg)
    seg=Image.fromarray(seg.astype('uint8'))
    seg.save(output_dir+'cs_'+fname,'JPEG')    
    plt.show()
    
    
def init_membership_mat(pixel_range,cls_range):
    mat = np.random.random((pixel_range,cls_range))
    rsum = np.sum(mat,axis=1)
    mat = mat /rsum[:,np.newaxis] 
    return mat


def generate_clusters(pix_range,count,clus_list):
    clusters=[]
    div_list=[]
    div_list.append(0)
    #print(clus_list)
    for k in range(len(clus_list)-1):
        div_list.append((clus_list[k]+clus_list[k+1])/2)
    div_list.append(256)
    div_list = list(map(int,div_list))
    #print(div_list)
    for k in range(len(div_list)-1):
        clusters.append(pix_range[div_list[k]: div_list[k+1]])
    #print(clusters)
    return clusters
    
def calc_mean(clusters,count,V,r):
    means=[]
    for j in range(len(clusters)):
        weights = np.array([count[int(i)] for i in clusters[j]])
        mem_v  =  np.array([V[int(i)] for i in clusters[j]])
        clus_mean = int(np.sum(clusters[j]*weights*np.power(mem_v[:,j],r))/np.sum(weights*np.power(mem_v[:,j],r)))
        means.append(clus_mean)
    return np.array(means)
        
        
def calc_membership_mat(pix_range,means,r):
    d_mat = np.zeros((pix_range.shape[0],means.shape[0]))
    V =  np.zeros(d_mat.shape)
    for j in range(len(means)):
        v_j= np.abs(pix_range - means[j])
        v_j[np.where(v_j == 0)] = 1
        d_mat[:,j]=v_j
    d_mat = np.power(d_mat,2/(r-1))
    
    for j in range(d_mat.shape[1]):
        p = np.sum(np.array([d_mat[:,j]/d_mat[:,k] for k in range(d_mat.shape[1])]),axis=0)
        V[:,j]=1/p
    return V
        
    
    
    
    
    
def fcm(pix_range,count,C):
    
   # V = init_membership_mat(pix_range.shape[0],C)
    means=np.array(random.sample(range(0, 255),C))
    V = None
    clusters=None
    err_tol = 1e-5
    iters=0
    error=1e5
    while(error > err_tol and iters<200):    
        means.sort()
        V = calc_membership_mat(pix_range,means,2)
        clusters = generate_clusters(pix_range,count,means)
        means_new = calc_mean(clusters,count,V,2)
        error = np.linalg.norm(means_new-means)
        means = means_new 
        iters+=1
    return [clusters,V]
    


def fuzzy_cmeans(fname,K=2,get_stat = False):   
    stime = time.time()
    arr_img=get_image(input_dir+fname)
    count,bins = np.histogram(arr_img,256,[0,256])
    clusters,V = fcm(bins,count,K) #fcm algorithm for 2 clusters
    seg = np.zeros(arr_img.shape)
    for i in range(arr_img.shape[0]):
    	for j in range(arr_img.shape[1]):
    			seg[i][j] = arr_img[i][j]*V[arr_img[i][j],1]*255
    etime = time.time()
    if get_stat:
        time_taken = etime-stime
        db_index =1 
        sqr_err = 1
        return (time_taken,db_index,sqr_err)
    else:
        show_results_fuzzy_cmeans(arr_img,seg,K,fname)
    


    