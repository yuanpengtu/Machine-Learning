# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:35:27 2018

@author: Amitesh863
"""

import random
import numpy as np
from matplotlib import pyplot as plt
from imgrs import get_image as gi


input_dir = '../data/'
output_dir = '../result/'

def kmeans(hist):
    meanc1=0
    meanc2=0
    rand=random.sample(range(0, 255),2)
    rand.sort()
    j=0
    while(1):
        if(j==0):
            c1=rand[0]
            c2=rand[1]
        else:
            c1=meanc1
            c2=meanc2
        cluster1=[]
        cluster2=[]
        pixel_count1=pixel_count2=0
        sum1=sum2=0
   
        for i,count in enumerate(hist):
            
            disti1=abs(i-c1)
            disti2=abs(i-c2)
            if(disti1<disti2):
           
                cluster1.append(i)
                pixel_count1+=count
                sum1+=i*count
            else:
       
                cluster2.append(i)
                pixel_count2+=count
                sum2+=i*count
       
        meanc1=int(sum1)/int(pixel_count1)
        meanc2=int(sum2)/int(pixel_count2)
        j+=1
     
        
        if(c1==meanc1 and c2==meanc2):
            return[cluster1,cluster2]
    return[cluster1,cluster2]




arr_img=gi(input_dir+"blob.jpg")


rows,columns = np.shape(arr_img)

hist,bins = np.histogram(arr_img,256,[0,256])
clusters = kmeans(hist)
seg = np.zeros((rows,columns))

for i in range(rows):
	for j in range(columns):
			if (arr_img[i][j] in clusters[1]):
				seg[i][j] = int(1)

			else:
				seg[i][j] = int(0)


plt.imshow(seg, cmap="gray")
plt.imsave(output_dir+"kmeans.png",seg, cmap="gray")
plt.show()
