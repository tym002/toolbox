#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:55:28 2019

@author: tianyu
"""

import numpy as np 
import os
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import math
import plotly
from sklearn.linear_model import LinearRegression

'''
This script makes figures for my paper 'Volumetric landmark detection with a 
multi-scale translation equivariant neural network'.

'''


def confidence_interval(data, confidence):
    '''
    compute the confidence interval for given data and confidence level[0,1]
    input: the data and confidence level 
    output: confidence interval [start,end]
    
    '''
    n = data.shape[0]
    m = np.mean(data)
    std_err = np.std(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h
    return start, end

def check_inside(point,x,y,z,a,b,c):
    '''
    check if a point is inside a ellipsoid 
    input: the point, center of the ellipoid, axis of the ellipsoid 
    output: boolean value indicating inside or not
    '''
    return (point[0,0] - x)**2/a**2 + (point[0,1]-y)**2/b**2 + (point[0,2]-z)**2/c**2 < 1


#load data 
path = '/Users/tianyu/mertlab/Bifurcation/paperdata/patch(0817b).csv'

data = pd.read_csv(path) 


# plot the violinplot
fig, ax = plt.subplots(figsize =(9, 7)) 
sns.violinplot(cut = 0,scale = "count",linewidth = 2,ax = ax , data = data.iloc[:, :] ) 
ax.set(xlabel='Architectures', ylabel='Euclidian Distance(mm)')




#visualization the image slice
X = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Images208_ori.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0703e.py',delimiter= ',')
N=33
c = 30
d = 20
img = X[N,:,:,:,0].astype('float32').copy()
yl = Y[N,0,0:3].copy()
ylp = Yp[N,0:3].copy()

img1 = np.swapaxes(img,0,2).copy()
cimg1 = img1[int(yl[2])-c:int(yl[2])+c,int(yl[1])-c:int(yl[1])+c,int(yl[0])]
print(cimg1.shape)
plt.imshow(cimg1,cmap="gray",origin="lower")
plt.plot(c,c, 'r*',markersize=8)
plt.plot(c+ylp[1]-yl[1],c+ylp[2]-yl[2], 'go',markersize=8)

plt.figure(2)
cimg2 = img[int(yl[0])-c:int(yl[0])+c,int(yl[1])-c:int(yl[1])+c,int(yl[2])]
plt.imshow(cimg2,cmap="gray",origin="lower")
plt.plot(c,c, 'r*',markersize=8)
plt.plot(c+ylp[1]-yl[1],c+ylp[0]-yl[0], 'go',markersize=8)

# visualize heatmap for our method 
H = np.load('/Users/tianyu/mertlab/Bifurcation/result/heatmap0703e.npy')
H1 = H[N,:,:,:,0].copy()
H2 = np.swapaxes(H1,0,2).copy()

plt.figure(3)
cimg3 = H2[int(yl[2])-c:int(yl[2])+c,int(yl[1])-c:int(yl[1])+c,int(yl[0])]
plt.imshow(cimg3,origin="lower")
plt.plot(c,c, 'r*',markersize=8)
plt.plot(c+ylp[1]-yl[1],c+ylp[2]-yl[2], 'go',markersize=8)

plt.figure(4)
cimg4 = H1[int(yl[0])-c:int(yl[0])+c,int(yl[1])-c:int(yl[1])+c,int(yl[2])]
plt.imshow(cimg4,origin="lower")
plt.plot(c,c, 'r*',markersize=8)
plt.plot(c+ylp[1]-yl[1],c+ylp[0]-yl[0], 'go',markersize=8)


# visualize gaussian heatmap 
H = np.load('/Users/tianyu/mertlab/Bifurcation/result/Prediction0827c.npy')
H1 = H[N,:,:,:,0].copy()
H2 = np.swapaxes(H1,0,2).copy()

plt.figure(5)
m5 = H2[int(yl[2])-c:int(yl[2])+c,int(yl[1])-c:int(yl[1])+c,int(yl[0])]
coor5 = np.where(m5 == np.amax(m5))
print(coor5)
i5 = plt.imshow(m5,origin="lower")
plt.plot(c,c, 'r+',markersize=8)
plt.plot(coor5[1][0],coor5[0][0], 'kx',markersize=8)

plt.figure(6)
m6 = H1[int(yl[0])-c:int(yl[0])+c,int(yl[1])-c:int(yl[1])+c,int(yl[2])]
i6 = plt.imshow(m6,origin="lower")
coor6 = np.where(m6 == np.amax(m6))
plt.plot(c,c, 'r+',markersize=8)
plt.plot(coor6[1][0],coor6[0][0], 'kx',markersize=8)

# plot all 6 figures together
f, axs = plt.subplots(2, 3)
axs[0,0].axis('off')
axs[0,0].imshow(cimg1,cmap="gray",origin="lower")
axs[0,0].plot(c,c, 'r+',markersize=8)
axs[0,0].plot(c+ylp[1]-yl[1],c+ylp[2]-yl[2], 'kx',markersize=8)
axs[0,0].title.set_text('(a)')

axs[1,0].axis('off')
axs[1,0].imshow(cimg2,cmap="gray",origin="lower")
axs[1,0].plot(c,c, 'r+',markersize=8)
axs[1,0].plot(c+ylp[1]-yl[1],c+ylp[0]-yl[0], 'kx',markersize=8)
axs[1,0].title.set_text('(b)')


axs[0,1].axis('off')
axs[0,1].imshow(cimg3,origin="lower")
axs[0,1].plot(c,c, 'r+',markersize=8)
axs[0,1].plot(c+ylp[1]-yl[1],c+ylp[2]-yl[2], 'kx',markersize=8)
axs[0,1].title.set_text('(c)')


axs[1,1].axis('off')
axs[1,1].imshow(cimg4,origin="lower")
axs[1,1].plot(c,c, 'r+',markersize=8)
axs[1,1].plot(c+ylp[1]-yl[1],c+ylp[0]-yl[0], 'kx',markersize=8)
axs[1,1].title.set_text('(d)')

axs[0,2].axis('off')
axs[0,2].imshow(m5,origin="lower")
axs[0,2].plot(c,c, 'r+',markersize=8)
axs[0,2].plot(coor5[1][0],coor5[0][0], 'kx',markersize=8)
axs[0,2].title.set_text('(e)')

axs[1,2].axis('off')
axs[1,2].imshow(m6,origin="lower")
axs[1,2].plot(c,c, 'r+',markersize=8)
axs[1,2].plot(coor6[1][0],coor6[0][0], 'kx',markersize=8)
axs[1,2].title.set_text('(f)')

# another way to plot them, heatmap overlay on top of the image 
fig, ax = plt.subplots(2,2)
ax[1,0].axis('off')
ax[1,0].imshow(cimg4,origin="lower",alpha = 0.3,zorder =2 )
ax[1,0].imshow(cimg2,cmap="gray",origin="lower",zorder = 1)
ax[1,0].plot(c,c, 'r+',markersize=8)
ax[1,0].plot(c+ylp[1]-yl[1],c+ylp[0]-yl[0], 'kx',markersize=6)
ax[1,0].title.set_text('(B)')

ax[0,0].axis('off')
ax[0,0].imshow(cimg3,origin="lower",alpha = 0.3,zorder =2 )
ax[0,0].imshow(cimg1,cmap="gray",origin="lower",zorder = 1)
ax[0,0].plot(c,c, 'r+',markersize=8)
ax[0,0].plot(c+ylp[1]-yl[1],c+ylp[2]-yl[2], 'kx',markersize=6)
ax[0,0].title.set_text('(A)')

ax[0,1].axis('off')
ax[0,1].imshow(m5,origin="lower",alpha = 0.3,zorder =2 )
ax[0,1].imshow(cimg1,cmap="gray",origin="lower",zorder = 1)
ax[0,1].plot(c,c, 'r+',markersize=8)
ax[0,1].plot(coor5[1][0],coor5[0][0], 'kx',markersize=6)
ax[0,1].title.set_text('(C)')

ax[1,1].axis('off')
ax[1,1].imshow(m6,origin="lower",alpha = 0.3,zorder =2 )
ax[1,1].imshow(cimg2,cmap="gray",origin="lower",zorder = 1)
ax[1,1].plot(c,c, 'r+',markersize=8)
ax[1,1].plot(coor6[1][0],coor6[0][0], 'kx',markersize=6)
ax[1,1].title.set_text('(D)')




#plot noise injection figures 
mask_dir = '/Users/tianyu/mertlab/Bifurcation/noise0814a/'
lst = []
arr = np.zeros((50, 54, 3))
for _ , _,files in os.walk(mask_dir):
    for f in files:
        if f.endswith('.py'):
            Y = np.loadtxt(fname = os.path.join(mask_dir,f),delimiter= ',' )
            lst.append(Y)


print(len(lst))
for i in range(0,len(lst)):
    arr[i,:,:] = lst[i]

arr = np.swapaxes(arr,0,1)
print(arr.shape)
print(arr[0])

y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0814a.py',delimiter= ',' )
# A is volume, which is y
A = []
# B is volume, which is x
B = []
std = []

conf = 0.9
count = 0 
for j in range(0,Yp.shape[0]):
    rst = arr[j]
    xdata = rst[:,0]
    ydata = rst[:,1]
    zdata = rst[:,2]
    x1,x2 = confidence_interval(xdata,conf**(1./3))
    y1,y2 = confidence_interval(ydata,conf**(1./3))
    z1,z2 = confidence_interval(zdata,conf**(1./3))
    xmean = (x1+x2)/2
    xrange = (x2-x1)/2
    ymean = (y1+y2)/2
    yrange = (y2-y1)/2
    zmean = (z1+z2)/2
    zrange = (z2-z1)/2
    volume = xrange*yrange*zrange*4*math.pi/3
    A.append(volume)
    ldis = np.sqrt(np.sum(np.square((y[j,0,0:3]-Yp[j])*0.5)))
    B.append(ldis)
    if check_inside(y[j],xmean,ymean,zmean,xrange,yrange,zrange):
        count = count + 1 
print(count/j)

st = np.std(arr,1)
stm = np.mean(st,1)
stm = np.reshape(stm,(54,1))
print(st)
print(stm)
for i in range(0,54):
    std.append(stm[i,0])

plt.figure(1)
plt.scatter(B,std)
plt.xlabel('distance')
plt.ylabel('Average STD')

x = np.array(B).reshape((-1, 1))
y = np.array(A)
plt.figure(2)
plt.scatter(x,y)
p = np.linspace(0,20,1000)
q = 1976.26143537*p + 16539.096852323113
plt.plot(p, q, '-r', label='y=1976x + 16539')
plt.xlabel('Euclidean Distance(mm)')
plt.ylabel('90% Confidence Volume(mm\u00b3)')
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)



#plot inclusion percentage vs. confidence level
icd = [0.962,0.906,0.868,0.792,0.755,0.698,0.604,0.491,0.415]
perc = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
p = np.linspace(0.4,0.9,100)
q = p
plt.plot(p, q, '-r', label='y=x')
plt.scatter(perc,icd)
plt.legend(loc='upper left')
plt.xlabel('Confidence Level')
plt.ylabel('Percentage of Inclusion')

