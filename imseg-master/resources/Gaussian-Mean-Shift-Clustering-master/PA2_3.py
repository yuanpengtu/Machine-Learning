
import matplotlib.pyplot as plt
import numpy as np


import cv2


'''
Mean shift algorithm considers the feature space as emprirically probability density
function. For each data point in the set, the algorithm associates it with nearby peak
of dataset probability function. The algorithm defines a window around the datapoint and 
calculates the mean of the datapoint accordingly. Then it shifts the centre of the window
to the mean calculated and this process keeps going on until the window converges. 

Basic Mean Shift Algorithm:
    a) Fix a window around the data point.
    b) Compute the mean within the window.
    c) Shift the window until it converges. 

For Gaussian Mean Shift,

    a) Guassian Kernel
        f(x) = exp((-x*x)/(2*s*s))
    b) Kernel density estimation based on Parzen window technique


'''
threshold = 0.01
hs = 20         #spatial resolution
hr = 35         #range resolution


#get the color image ready for input
Image1 = 'input3.jpg'
Im = cv2.imread(Image1)
I = np.array(Im)

#split the color image into its three channels of RGB
B,G,R = cv2.split(Im)


#we need to select a window so to preprocess the image, we pad the image with the image width and image height so the window doesn't go out of bounds
R_pad= np.pad(R,((R.shape[0],R.shape[0]),(R.shape[1],R.shape[1])),'symmetric')
G_pad= np.pad(G,((G.shape[0],G.shape[0]),(G.shape[1],G.shape[1])),'symmetric')
B_pad= np.pad(B,((B.shape[0],B.shape[0]),(B.shape[1],B.shape[1])),'symmetric')

#initializing the look up table for the gaussian weight
weight = np.linspace(0, 256*256, num=256*256+1)
v = hr*hr
t = np.divide(weight,v)

weight = np.exp(-t)
print ("It takes about 60 iterations to converge. It takes about 7 minutes to produce the results. ")
iter = 0
while True:

    iter+=1
    totalDen  = np.zeros(R.shape,dtype=int);
    totalNumR = np.zeros(R.shape,dtype=int);
    totalNumG = np.zeros(R.shape,dtype=int);
    totalNumB = np.zeros(R.shape,dtype=int);
    squaredifR = np.zeros(R.shape,dtype=int)
    squaredifG = np.zeros(R.shape,dtype=int)
    squaredifB = np.zeros(R.shape,dtype=int)

    #creating the window and looking at each pixel within -hs and hs
    for i in range(-hs,hs):
        for j in range(-hs,hs):
            RNew = R_pad[R.shape[0]+i-1:(2*R.shape[0])+i-1,R.shape[1]+j-1:(2*R.shape[1])+j-1]
            GNew = G_pad[G.shape[0]+i-1:(2*G.shape[0])+i-1,G.shape[1]+j-1:(2*G.shape[1])+j-1]
            BNew = B_pad[B.shape[0]+i-1:(2*B.shape[0])+i-1,B.shape[1]+j-1:(2*B.shape[1])+j-1]

            #computing the distance square arrays based on the original arrays and also add 1 to avoid minimum value of zero for each array
            difference_square = np.multiply((R-RNew),(R-RNew))
            squaredifR = np.multiply((R-RNew).astype(int),(R-RNew).astype(int))
            squaredifG = np.multiply((G-GNew).astype(int),(G-GNew).astype(int))
            squaredifB = np.multiply((B-BNew).astype(int),(B-BNew).astype(int))

            #to avoid minimum value of zero for each R,G,B array
            squaredifR = squaredifR+1
            squaredifG = squaredifG+1
            squaredifB = squaredifB+1

            #get the kernel weight from weight
            WeightMapR = weight[squaredifR]
            WeightMapG = weight[squaredifG]
            WeightMapB = weight[squaredifB]

            #compute the product of 3 weight matrices since they are  e (x-xn)*(x-xn) +(y-yn)*(y-yn)+(z-zn)*(z-zn) =e (x-xn)*(x-xn) * e(y-yn)*(y-yn)* e(z-zn)*(z-zn)

            temp = np.multiply(WeightMapR,WeightMapG)
            WeightFinal = np.multiply(temp,WeightMapB)


            totalDen = totalDen+WeightFinal

            totalNumR = totalNumR+(RNew*WeightFinal)
            totalNumG = totalNumG+(GNew*WeightFinal)
            totalNumB = totalNumB+(BNew*WeightFinal)

    RFinal  = np.divide(totalNumR,totalDen)
    GFinal  = np.divide(totalNumG,totalDen)
    BFinal  = np.divide(totalNumB,totalDen)

    RVector = np.around(RFinal)-np.round(R)
    GVector = np.around(GFinal)-np.round(G)
    BVector = np.around(BFinal)-np.round(B)

    R = np.around(RFinal)
    G = np.around(GFinal)
    B = np.around(BFinal)

    #if final R,G,B values is less than the threshold then we can break the while loop
    Mean = np.mean(RVector+GVector+BVector)
    if abs(Mean)<threshold:
        break;
   
    plt.title('iteration'+str(iter))
    
    print('iteration no is ',iter)
    print('shifted mean value in this iteraton',abs(Mean))

OutputImage = np.zeros([R.shape[0],R.shape[1],3])
OutputImage[:,:,0] = B
OutputImage[:,:,1] = G
OutputImage[:,:,2] = R
plt.figure()

l=np.array(OutputImage,dtype="uint8")
plt.figure()
plt.imshow(l)
plt.show()


