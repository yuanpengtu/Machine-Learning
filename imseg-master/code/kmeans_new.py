# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 23:32:53 2018

@author: durgesh singh
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time



input_dir = '../data/'
output_dir = '../result/'


def show_results_kmeans(orig,seg,K,fname):
    plt.figure()
    ax1 =plt.subplot(121)
    ax2 =plt.subplot(122)
    ax1.set_title('Original Image')
    ax1.imshow(orig)
    ax2.set_title('Result of segmented image \nwith K = {}'.format(K))
    ax2.imshow(seg)
    seg=Image.fromarray(seg.astype('uint8'))
    seg.save(output_dir+'ks_'+fname,'JPEG')    
    plt.show()
    
    
def assign_cluster(data_arr,mediods):
    #mediods = data_arr[med_indx]
    dists = np.zeros((data_arr.shape[0],len(mediods)))
    
    for i in range(mediods.shape[0]):
        diff = data_arr - mediods[i]
        dists[:,i] = np.linalg.norm(diff,axis=1)
    cluster_nos= np.argmin(dists,axis=1).reshape(data_arr.shape[0],1)
    #print(cluster_nos)
    data_arr = np.concatenate([data_arr,cluster_nos],axis=1)
    return data_arr
    
    


def calc_cost(clus_data_arr,means):
    cost =0
    for j in range(means.shape[0]): 
        clus =clus_data_arr[np.where(clus_data_arr[:,-1] == j)][:,:-1]
        mean=means[j]
        cost+=np.sum(np.linalg.norm(clus -mean,axis=1))
    return cost
        
    
    
    
def calc_mean(clus_data_arr,means):
    #mediods= data_arr[med_indx]
    new_means =[]
    for j in range(means.shape[0]): 
        clus =clus_data_arr[np.where(clus_data_arr[:,-1] == j)][:,:-1]
        new_mean = np.mean(clus,axis=0)
        new_means.append(new_mean)
    new_means = np.array(new_means)
    return new_means

    

def k_means(data_arr,k,get_stat=False):
    iters=1
    mean_indx = np.random.choice(data_arr.shape[0],k,replace=False) 
    means = data_arr[mean_indx]
    clus_data_arr=None
    cost_arr =[]
    
    while(iters<200):
        clus_data_arr = assign_cluster(data_arr,means)
        means_new = calc_mean(clus_data_arr,means)
        if (np.array_equal(means,means_new)):
            break
        means = means_new
        #print(mediods)
        if(get_stat):    
            cost = calc_cost(clus_data_arr,means)
            #print(cost)
            cost_arr.append((iters,cost))
        iters+=1  
        
      
    if get_stat:
        return (clus_data_arr,cost_arr,means)
    return clus_data_arr






def kmeans(fname,K=2,get_stat = False): 
    stime = time.time()
    arr_img= np.array(Image.open(input_dir+fname).convert('L').resize((256,256)))    
    #handling grayscale vs color case
    if (len(arr_img.shape) == 2):
        w,h = arr_img.shape
        data_arr = arr_img.reshape(w*h,1)
    elif(len(arr_img.shape) == 3):
        w,h,d = arr_img.shape
        data_arr = arr_img.reshape((w*h,d))  
    
    clus_data_arr,c,final_means = k_means(data_arr,K,get_stat=True)
    
    #for each mean and its corresponding cluster
    for k in range(final_means.shape[0]):
        idx = np.where(clus_data_arr[:,-1] == k)
        clus_data_arr[idx,:-1]=final_means[k]
        
    #showing the final clustered image
    data_arr = clus_data_arr[:,:-1]
    data_arr= data_arr.reshape(arr_img.shape)
    etime = time.time()
    if get_stat:
        time_taken = etime-stime
        db_index = 1
        sqr_err = 0
        return (time_taken,db_index,sqr_err)
    else:    
        show_results_kmeans(arr_img,data_arr,K,fname)
    

