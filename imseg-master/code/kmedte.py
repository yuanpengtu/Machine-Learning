# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:56:24 2018

@author: Amitesh863
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

def random_color(p):
   COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c']
   return COLORS[p]

def plotdata(result,k,cluster):
   for p in range(k):
       for i in range(0,len(result)):
           if(cluster[i]==p):
               plt.scatter(result[i,0],result[i,1],c=random_color(p+1))

def k_means(data,u_array,k):
   dist_from_mean=np.linalg.norm(np.subtract(data,u_array[0]),axis=1)
   #print(dist_from_mean)
   cluster=np.zeros((data.shape[0],1))
   for i in range(1,k):
       #print(i)
       new_dist=np.linalg.norm(np.subtract(data,u_array[i]),axis=1)
       boolean=np.greater(dist_from_mean,new_dist)
       index_where_greater=np.where(boolean)
       for j in index_where_greater:
           cluster[j]=i
           dist_from_mean[j]=new_dist[j]
   return cluster

def euclidean(data,u_array):
   return np.linalg.norm(np.subtract(data,u_array))

def distance_bet_pts(result,datapt):
   med_distance=np.linalg.norm(np.subtract(result,datapt))
   return med_distance

def getdata():
   reader = csv.reader(open("data2.csv", "rt"), delimiter=",")
   x = list(reader)
   result = np.array(x).astype("float")
   k=3
   idx = np.arange(result.shape[0])
   selected = np.random.choice(idx, k, replace=False)

   med_array=[]
   for i in range(len(selected)):
       med_array.append(result[selected[i]])

   cluster=k_means(result,med_array,k)
   iterator=0
   i=0
   while(1):
       cluster_i=np.where(cluster==i)
       l1=np.ndarray.tolist(cluster_i[0])
       xij=[]
       for p in l1:
           xij.append(result[p])
       med_distance=distance_bet_pts(xij,med_array[i])
       min_dist=np.inf
       for p in l1:
           p_distance=distance_bet_pts(xij,result[p])
           if(p_distance<min_dist):
               min_dist=p_distance
               min_index=p
       #print(min_index)
       mediod_old=np.copy(med_array)
       if(min_dist<med_distance):
           med_array[i]=result[min_index]
           cluster=k_means(result,med_array,k)
       i=(i+1)%k
       flag=[]
       for array in range(len(mediod_old)):
           med_new=np.copy(med_array)
           mediod_old=np.array(mediod_old)
           med_new=np.array(med_new)
           print('"{}"'.format(med_new))
           print('"{}"'.format(mediod_old))
           if(np.array_equal(med_new[array],mediod_old[array])):
               flag.append(True)
           else:
               flag.append(False)
       #print(flag)
       if(all(element for element in flag)):
           break
       iterator=iterator+1
   plt.figure()
   plotdata(result,k,cluster)

getdata()