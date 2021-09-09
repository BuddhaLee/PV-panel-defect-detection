import cv2
import numpy as np
import os

def Gray_img(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    #cv2.imshow('gray', gray)
    #cv2.imwrite('gray.png', gray)

    return gray

def Exteact_temp(img):
    gray_img = Gray_img(img)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurshape=blur.shape
    #grayvalue=0
    grayvaluemax=0
    for i in range(blurshape[0]):
        for j in range(blurshape[1]):
            grayvalue = blur[i,j]#求灰度值
            if grayvalue >grayvaluemax:
                grayvaluemax=grayvalue
    print(grayvaluemax)
    a=int(grayvaluemax)*int(grayvaluemax)*0.0021
    print(a)
    b=-0.35*grayvaluemax
    c=40.53
    temp=a+b+c
    return temp

imgpa="/home/lee/hainan/hainan_red/ssd/hainan/区域1红外/DJI_20200724102724_0379_THRM.JPG"
img=cv2.imread(imgpa)
te=Exteact_temp(img)
print(te)
