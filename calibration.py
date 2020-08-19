# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:00:46 2020

@author: Hana Luo
"""

import numpy as np
import cv2 as cv
import glob
# 设置终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# 用于存储所有图像的对象点和图像点的数组。
objpoints = [] # 真实世界中的3d点
imgpoints = [] # 图像中的2d点
images = glob.glob('E:/Project_Stereo_left/left/*.jpg')

for fname in images:
    #对每张图片，找到角点并记录其世界坐标和图像坐标
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 找到棋盘角落，将角点存在corners中；ret为flag，判断是否找到角点
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # 如果找到，添加对象点，图像点（细化之后）
    if ret == True:
        objpoints.append(objp)
        # 进行亚像素级角点检测，提高精度
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # 绘制并显示拐角
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        #cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#打印矩阵参数
'''
每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组
mtx，相机内参；dist，畸变系数；revcs，旋转矩阵；tvecs，平移矩阵。
'''
print ("newcameramtx内参:\n",newcameramtx)
print ("dist畸变值:\n",dist) # distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print ("newcameramtx旋转（向量）外参:\n",rvecs)
print ("newcameramtx平移（向量）外参:\n",tvecs)
#计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    tot_error += error
 
print ("total error: ", tot_error/len(objpoints))
 
