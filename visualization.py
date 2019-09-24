#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:00:30 2019

@author: tianyuma
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
from keras import backend as K
import tensorflow as tf
from scipy import ndimage 
import statistics
    
# change the view of the image and print all the stacks  
'''
A = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Images448_ori.npy')
B = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori_p.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0903b_p.py',delimiter= ',' )
print(Yp.shape)
N = 45
A1 = A[N].copy()  
A2 = np.copy(A1[:,:,:,-1])
B1 = B[N].copy()
B2 = B1[0,0:3].copy()
ypl = Yp[N,0:3].copy()
print(ypl)
A2 = np.swapaxes(A2,0,2).copy()
A2 = A2.transpose(1,0,2).copy()
A2 = A2.astype('float32')
a = 2
b = 1
c = 0
x = B2[a]
y = B2[b]
z = B2[c]
xp = ypl[a]
yp = ypl[b]
zp = ypl[c]
for i in range (max(0,int(z-10)),min(A2.shape[2],int(z+10))):
    print("ground truth is: ",z)
    print("Prediction is: ",zp)
    plt.figure(i)
    plt.imshow(A2[:,:,i],cmap="gray",origin="lower")
    plt.plot(x,y,'ro')
    plt.plot(xp,yp,'go')
    plt.title(i)
 '''   

#visualize all image
'''
X = np.load('/Users/tianyu/mertlab/Bifurcation/Save0721/Test_images416_ori.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0721/Test_Mask_Coordinates416_ori.npy')
print("train_image shape is: ", X.shape)
print("train_mask shape is: ", Y.shape)
print(X.dtype)
print(Y.dtype)

for i in range (0,Y.shape[0]):
    N = i
    X1 = X[N]
    X2 = X1[:,:,:,-1]
    X2 = X2.transpose(1,0,2)
    plt.figure(i)

    Y1 = Y[N]

    Y1 = Y1[0]

    yl = Y1[0:3]

    #yr = Y1[3:6]

    X2 = X2.astype('float32')
    plt.imshow(X2[:,:,int(yl[2])],cmap="gray",origin="lower")
    plt.plot(yl[0],yl[1], 'ro')
    #plt.figure(2)
    #plt.imshow(X2[:,:,int(yr[2])],cmap="gray",origin="lower")
    #plt.plot(yr[0],yr[1],'go')
    plt.title(i)
    #print(i, "th has ", "left z: ", yl[2], "right z: ", yr[2])
print("max value is:",np.amax(X))
print("min value is:",np.amin(X))
print("avg value is:",np.mean(X)) 
'''

#visualize one side of image 
'''
img = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Train_images208_ori.npy')
A = np.load('/Users/tianyu/mertlab/Bifurcation/result/Prediction0827c_es.npy')
B = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
print("train_image shape is: ", A.shape)
print("train_mask shape is: ", B.shape)
h = []
for i in range (0,B.shape[0]):
    N = i
    X1 = A[N].copy()
    X2 = X1[:,:,:,-1].copy()
    plt.figure(i)
    #print("train_image shape is: ", Y.shape)
    Y1 = B[N,0,:]
    X2 = X2.transpose(1,0,2).astype('float32')
    plt.imshow(X2[:,:,int(Y1[2])],cmap="gray",origin="lower")
    plt.plot(Y1[0],Y1[1], 'ro')
    plt.title(i)
    coor = np.where(X2 == np.amax(X2))
    coor = np.reshape(np.asarray(coor),(1,3))
    #print(coor.shape)
    print("max value is:",np.amax(A[i]))
    print("min value is:",np.amin(A[i]))
    print("avg value is:",np.mean(A[i]))
    h.append(np.sqrt(np.sum(np.square((coor-B[i,:,0:3])*1))))
print(np.mean(h))
for i in range(0,B.shape[0]):
    print(i)
    #print("max value is:",np.amax(A[i]))
    #print("min value is:",np.amin(A[i]))
    #print("avg value is:",np.mean(A[i]))
    #print(np.amax(A))
    coor = np.where(A == np.amax(A[i]))
    print(coor)
    #print(A[i,:,coor[2][0],coor[3][0],0])
    plt.imshow(A[i,:,:,coor[3][0],0],cmap="gray",origin="lower")
'''


# visualize prediction and ground truth
'''
X = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Images448_ori.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori_p.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0903b_p.py',delimiter= ',' )
print(Yp.shape)
p = 0.5
N = 45
print(Y[N,0,2])
#print(Y[N,0,5])
print(Yp[N,2])
#print(Yp[N,5])
X1 = X[N+2].copy()
X2 = X1[:,:,:,-1].copy()
X2 = X2.transpose(1,0,2).copy()
X2 = X2.astype('float32')
#print("train_image shape is: ", Y.shape)
Y1 = Y[N].copy()
#print(Y1)
Y1 = Y1[0].copy()
#print("Y1 is " ,Y1)
yl = Y1[0:3].copy()
print("yl is ",yl)
#yr = Y1[3:6].copy()
#print("yr is",yr)

ypl = Yp[N,0:3].copy()
print("ypl is ",ypl)
#ypr = Yp[N,3:6].copy()
#print("ypr is ",ypr)

ldis = np.sqrt(np.sum(np.square((yl-ypl)*p)))
print("left point distance is: ", ldis)

#rdis = K.sqrt(K.sum(K.square(yr-ypr)))
#print("Right point distance is: ", K.eval(rdis))


plt.figure(1)
plt.imshow(X2[:,:,int(yl[2])],cmap="gray",origin="lower")
plt.plot(yl[0],yl[1], 'ro')
plt.plot(ypl[0],ypl[1], 'go')
plt.title(yl[2])

plt.figure(2)
plt.imshow(X2[:,:,int(ypl[2])],cmap="gray",origin="lower")
plt.plot(ypl[0],ypl[1], 'go')
plt.title(ypl[2])

plt.figure(3)
plt.imshow(X2[:,:,int(yr[2])],cmap="gray",origin="lower")
plt.plot(yr[0],yr[1],'ro')
plt.title(yr[2])

plt.figure(4)
plt.imshow(X2[:,:,int(ypr[2])],cmap="gray",origin="lower")
plt.plot(ypr[0],ypr[1], 'go')
plt.title(ypr[2])
'''


# visualize prediction and ground truth in different perspective 
'''
X = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Images208_ori.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0806a_s1.py',delimiter= ',' )
print(Yp.shape)
N = 31

X2 = X[N,:,:,:,0].copy()
#X2 = X2.transpose(1,0,2).copy()
#print("train_image shape is: ", Y.shape)
Y1 = Y[N,0,:]
yl = Y1[0:3]
print("yl is ",yl)
#yr = Y1[3:6]
#print("yr is",yr)

ypl = Yp[N,0:3]
print("ypl is ",ypl)
#ypr = Yp[N,3:6]
#print("ypr is ",ypr)

ldis = K.sqrt(K.sum(K.square(yl-ypl)))
print("left point distance is: ", K.eval(ldis))

#rdis = K.sqrt(K.sum(K.square(yr-ypr)))
#print("Right point distance is: ", K.eval(rdis))

plt.figure(1)
x3 = X2[int(ypl[0]),:,:].transpose(1,0).copy()
plt.imshow(x3,cmap="gray",origin="lower")
plt.plot(yl[1],yl[2], 'ro')
plt.plot(ypl[1],ypl[2], 'go')
plt.title(ypl[0])

plt.figure(2)
plt.imshow(X2[int(ypl[0]),:,:],cmap="gray",origin="lower")
plt.plot(ypl[2],ypl[1], 'go')
plt.title(ypl[0])

plt.figure(3)
x4 = X2[int(yr[0]),:,:].transpose(1,0).copy()
plt.imshow(x4,cmap="gray",origin="lower")
plt.plot(yr[1],yr[2], 'ro')
plt.plot(ypr[1],ypr[2], 'go')
plt.title(yr[0])

plt.figure(4)
x5 = X2[:,int(ypr[0]),:].transpose(1,0).copy()
plt.imshow(x5,cmap="gray",origin="lower")
plt.plot(ypr[1],ypr[2], 'go')
plt.title(ypr[0])
'''

#compute average distance for one side test set 
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori_p.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0903d_es.py',delimiter= ',' )
l = Y.shape[0] 
p = 0.5
tol = 5
crt = 0
print(l)
print(Yp.shape)
ld = 0.0
for i in range (0,l):
    Y1 = Y[i]
    Y1 = Y1[0]
    yl = Y1[0:3]
    ypl = Yp[i,0:3]
    ldis = K.sqrt(K.sum(K.square((yl-ypl)*p)))
    ld = ld + ldis
    if K.eval(ldis)  <= tol:
        crt += 1 
ald = ld / l

print("left point average distance is: ", K.eval(ald))
print("accuracy with tol = ", tol, "is: ", crt/l)
'''

#compute average distance for left and right bifurcations of test set 
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0806b_s1.py',delimiter= ',' )
l = Y.shape[0] 
p = 1
tol = 5
crt = 0
print(l)
print(Yp.shape)
ld = 0.0
rd = 0.0
for i in range (0,l):
    Y1 = Y[i]
    Y1 = Y1[0]
    yl = Y1[0:3]
    yr = Y1[3:6]
    ypl = Yp[i,0:3]
    ypr = Yp[i,3:6]
    ldis = K.sqrt(K.sum(K.square((yl-ypl)*p)))
    rdis = K.sqrt(K.sum(K.square((yr-ypr)*p)))
    ld = ld + ldis
    rd = rd + rdis
    if ((K.eval(ldis) + K.eval(rdis))/2 <= tol):
        crt += 1 
ald = ld / l
ard = rd / l
ad = (ld + rd)/2/l
print("left point average distance is: ", K.eval(ald))
print("right point average distance is: ", K.eval(ard))
print("total average distance is: ", K.eval(ad))
print("accuracy with tol = ", tol, "is: ", crt/l)
'''

#compute average distance in x, y, and z direction 
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates112_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0814a_s2.py',delimiter= ',' )
l = Y.shape[0] 
p = 2
tol = 3
crt = 0
zrt = 0
t = 10
print(l)
print(Yp.shape)
xl = 0.0
yl = 0.0
#rd = 0.0
zl = 0.0
xx = 0.0
yy = 0.0
zz = 0.0
dd = 0.0
#zr = 0.0
for i in range (0,l):
    xl = Y[i,0,0].copy()
    yl = Y[i,0,1].copy()
    zl = Y[i,0,2].copy()
    #zzr = Y1[5]
    px = Yp[i,0].copy()
    py = Yp[i,1].copy()
    pz = Yp[i,2].copy()
    #ypr = Yp[i,3:5]
    #zpl = Yp[i,2]
    #zpr = Yp[i,5]
    xdis = abs((xl-px)*p)
    ydis = abs((yl-py)*p)
    zdis = abs((zl-pz)*p)
    adis = math.sqrt(xdis*xdis + ydis*ydis + zdis*zdis)
    #rdis = K.sqrt(K.sum(K.square((yr-ypr)*p)))
    #zldis = abs((zzl-zpl)*p)
    #zrdis = abs((zzr-zpr)*p)
    xx = xx + xdis
    #rd = rd + rdis
    yy = yy + ydis
    zz = zz + zdis
    #zr = zr + zrdis
    dd = dd + adis
    #if ((K.eval(ldis) + K.eval(rdis))/2 <= tol):
        #crt += 1 
    #if ((zldis+zrdis)/2 <= t):
        #zrt += 1 
#ald = ld / l
#ard = rd / l
#ax = (xd + rd)/2/l

#az = (zl+zr)/2/l

#print("total average distance in x and y is: ", K.eval(ad))
#print("accuracy in x-y with tol = ", tol, "is: ", crt/l)
#print("total average distance in z is: ", az)
#print("accuracy in z with tol = ", t, "is: ", zrt/l)
print(xx/l)
print(yy/l)
print(zz/l)
print(dd/l)
'''


# visualize heatmap
'''
V = np.load('/Users/tianyu/mertlab/Bifurcation/result/heatmap0616g.npy')
print("train_image shape is: ", V.shape)
N = 3
X1 = V[N]
X2 = X1[:,:,:,1]
X2 = X2.transpose(1,0,2)

for i in range (0,128):
    if (i%10 == 0):
        plt.figure(i)
        plt.imshow(X2[:,:,i],origin="lower",vmin = np.amin(X2),vmax = np.amax(X2))
        plt.title(i)
        
j = 15
plt.imshow(X2[:,:,j],origin="lower",vmin = np.amin(X2),vmax = np.amax(X2))
'''

# delete certain images
'''
save_path = '/Users/tianyu/mertlab/Bifurcation/Save0814/'
#cx = [210,206,199,192,165,161,159,153,149,147,140,138,131,126,125,119,100,97,96,84,77,67,66,64,56,53,23,21,19,5,3]
#cx = [50,43,41,35,23,21,14,7,4]
#cx = [158,157,154,153,147,131,120,117,115,81,63,39,37]
cx = [31,5]
X = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Images208_ori.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
print(len(cx))

for i in range(0,len(cx)):
    print("i is: ",i)
    j = cx[i]
    print("j is: ",j)
    X = np.delete(X, j,0)
    Y = np.delete(Y, j,0)
    print(X.shape)
    print(Y.shape)

np.save(os.path.join(save_path,'Test_Images208_ori_p'), X)
np.save(os.path.join(save_path,'Test_Mask_Coordinates208_ori_p'), Y)
'''

#visualize both ground truth and changed 
'''
X = np.load('/Users/tianyu/mertlab/Bifurcation/pydata/Train_Images.npy')
XN = np.load('/Users/tianyu/mertlab/Bifurcation/pydata/Train_Images_aug.npy')
Y = np.load('/Users/tianyu/mertlab/Bifurcation/pydata/Train_Mask_Coordinates.npy')
YN = np.load('/Users/tianyu/mertlab/Bifurcation/pydata/Train_Mask_Coordinates_aug.npy')
print("train_image shape is: ", X.shape)
print("train_mask shape is: ", Y.shape)

for i in range (0,X.shape[0]):
    N = i
    X1 = X[N]
    X2 = X1[:,:,:,-1]
    X2 = X2.transpose(1,0,2)
    
    XN1 = XN[N]
    XN2 = XN1[:,:,:,-1]
    XN2 = XN2.transpose(1,0,2)
    #print("train_image shape is: ", Y.shape)
    Y1 = Y[N]
    #print(Y1)
    Y1 = Y1[0]
    #print("Y1 is " ,Y1)
    yl = Y1[0:3]
    #print("yl is ",yl)
    yr = Y1[3:6]
    #print("yr is",yr)
    #print(X2.shape)
    YN1 = YN[N]
    #print(Y1)
    YN1 = YN1[0]
    #print("Y1 is " ,Y1)
    yNl = YN1[0:3]
    #print("yl is ",yl)
    yNr = YN1[3:6]

    X2 = X2.astype('float32')
    #plt.imshow(X2[:,:,5])
    plt.figure(2*i)
    plt.imshow(X2[:,:,int(yl[2])],cmap="gray",origin="lower")
    plt.plot(yr[0],yr[1],'ro')
    plt.figure(2*i+1)
    plt.imshow(XN2[:,:,int(yNl[2])],cmap="gray",origin="lower")
    plt.plot(yNr[0],yNr[1],'go')
    plt.title(i)
'''

#print all the difference, plot a histogram  
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0703e.py',delimiter= ',' )
print(Yp[47])
h = []
p = 1
l = Y.shape[0] 
tol = 2
crt = 0
ld = 0.0
rd = 0.0
for i in range (0,l):  
    Y1 = Y[i]
    Y1 = Y1[0]
    yl = Y1[0:3]
    #yr = Y1[3:6]
    ypl = Yp[i,0:3]
    #ypr = Yp[i,3:6]
    ldis = np.sqrt(np.sum(np.square((yl-ypl)*p)))
    #rdis = np.sqrt(np.sum(np.square((yr-ypr)*p)))
    #adis = (ldis + rdis)/2
    h.append(ldis)
    #print(i,"th image has left dis = ",np.round(ldis),"right dis = ", np.round(rdis),"average dis = ",np.round(adis))
    print(i,"th image has left dis = ",ldis)
    if i == 5:
        print(yl)
        print(ypl)
print("std is:",statistics.stdev(h))
print("mean is:",np.mean(h))
plt.hist(h,bins = 12)
np.savetxt('/Users/tianyu/mertlab/Bifurcation/paperdata/unetgaussian(0827c).csv', h,delimiter=' ')
'''

#print all the difference in x-y direction, plot a histogram
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0817c_p.py',delimiter= ',' )
hx = []
hy = []
hz = []
p = 0.5
l = Y.shape[0] 
ld = 0.0
rd = 0.0
for i in range (0,l):  
    Y1 = Y[i].copy()
    Y1 = Y1[0].copy()
    yl = Y1[0:3].copy()
    #yr = Y1[3:6].copy()
    ypl = Yp[i,0:3].copy()
    #ypr = Yp[i,3:6].copy()
    xldis = np.abs((yl[0]-ypl[0])*p)
    #xrdis = (yr[0]-ypr[0])*p
    yldis = np.abs((yl[1]-ypl[1])*p)
    #yrdis = np.sqrt(np.sum(np.square((yr[1]-ypr[1])*p)))
    zldis = np.abs((yl[2]-ypl[2])*p)
    #zrdis = np.sqrt(np.sum(np.square((yr[2]-ypr[2])*p)))
    #adis = (ldis + rdis)/2
    hx.append(xldis)
    hy.append(yldis)
    hz.append(zldis)
    if xldis>6:
        print("x = ",i)
    if yldis>10:
        print("y = ",i)
    if zldis>10:
        print("z = ",i)
    #print(i,"th image has left dis = ",np.round(ldis),"right dis = ", np.round(rdis),"average dis = ",np.round(adis))
    print(i,"th image has left dis = ",np.round(xldis))
plt.figure(1)
plt.hist(hx,bins = 12)
plt.figure(2)
plt.hist(hy,bins = 12)
plt.figure(3)
plt.hist(hz,bins = 12)
'''

# computer distance in mm
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0617e.py',delimiter= ',' )
l = Y.shape[0]
tol = 7
tol2= 5
crt = 0
crt2 = 0
print(l)
print(Yp.shape)
ld = 0.0
rd = 0.0
ld2 = 0.0
rd2 = 0.0
for i in range (0,l):
    Y1 = Y[i]
    Y1 = Y1[0]
    yl = Y1[0:3]
    yr = Y1[3:6]
    ypl = Yp[i,0:3]
    ypr = Yp[i,3:6]
    ldis = K.sqrt(K.sum(K.square(yl-ypl)))
    rdis = K.sqrt(K.sum(K.square(yr-ypr)))
    
    lx = (Y1[0] - Yp[i,0])*Y1[6]*4
    ly = (Y1[1] - Yp[i,1])*Y1[7]*4
    lz = (Y1[2] - Yp[i,2])*Y1[8]*4
    ll = np.sqrt(lx*lx + ly*ly + lz*lz)
    rx = (Y1[3] - Yp[i,3])*Y1[6]*4
    ry = (Y1[4] - Yp[i,4])*Y1[7]*4
    rz = (Y1[5] - Yp[i,5])*Y1[8]*4
    rr = np.sqrt(rx*rx + ry*ry + rz*rz)
    
    ld = ld + ldis
    rd = rd + rdis
    ld2 = ld2 + ll
    rd2 = rd2 + rr
    if ((K.eval(ldis) + K.eval(rdis))/2 <= tol):
        crt += 1 
    if ((ll+rr)/2 <= tol2):
        crt2 += 1
ald = ld / l
ard = rd / l
ad = (ld + rd)/2/l

ald2= ld2 / l
ard2 = rd2 / l
ad2= (ld2 + rd2)/2/l
print("left point average distance is: ", K.eval(ald))
print("right point average distance is: ", K.eval(ard))
print("total average distance is: ", K.eval(ad))
print("accuracy with tol = ", tol, "is: ", crt/l)
print("--------------------------------------------")

print("left point average distance is: ", ald2)
print("right point average distance is: ", ard2)
print("total average distance is: ", ad2)
print("accuracy with tol = ", tol2, "is: ", crt2/l)
'''

#print all the difference in mm, plot a histogram  
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448_ori.npy')
Yp = np.loadtxt(fname = '/Users/tianyu/mertlab/Bifurcation/result/Prediction0814a.py',delimiter= ',' )
h = []
l = Y.shape[0] 
tol = 2
crt = 0
ld = 0.0
rd = 0.0
for i in range (0,l): 
    Y1 = Y[i]
    Y1 = Y1[0]
    lx = (Y1[0] - Yp[i,0])*Y1[6]*4
    ly = (Y1[1] - Yp[i,1])*Y1[7]*4
    lz = (Y1[2] - Yp[i,2])*Y1[8]*4
    ll = np.sqrt(lx*lx + ly*ly + lz*lz)
    rx = (Y1[3] - Yp[i,3])*Y1[6]*4
    ry = (Y1[4] - Yp[i,4])*Y1[7]*4
    rz = (Y1[5] - Yp[i,5])*Y1[8]*4
    rr = np.sqrt(rx*rx + ry*ry + rz*rz)
    adis = (rr + ll)/2
    h.append(adis)
    print(i,"th image has left dis = ",ldis,"right dis = ", rdis,"average dis = ",adis)
plt.hist(h,bins = 30)
'''

#change the order of the 6th value 
'''
Y = np.load('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448.npy')
print(Y[6,0,0:6])
Y1 = Y[6].copy()
Y1 = Y1[0].copy()
s1 = Y1[0]
s2 = Y1[1]
s3 = Y1[2]

ypr = Y1[3:6].copy()
print("ypr is: ",ypr)

Y[6,0,0:3] = ypr
Y[6,0,3:6] = [s1,s2,s3]
print(Y[6,0,0:6])
#print("ypl is: ",ypl)
Y = np.save('/Users/tianyu/mertlab/Bifurcation/Save0814/Test_Mask_Coordinates448.npy',Y)
'''

# observe 208 and 416 images and ground truth
'''
Y1 = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates208.npy')
Y2 = np.load('/Users/tianyu/mertlab/Bifurcation/Save0608/Test_Mask_Coordinates416.npy')
print(Y1[8,0,0:6]*2)
print(Y2[8,0,0:6])
'''
