# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:04:48 2020

@author: Hana Luo
"""
import numpy as np
import cv2
import time
import cuda

'''
从KITTI数据集的标定文件calib_cam_to_cam中导入参数
标号2,3为彩色相机
'''
#双目相机参数

#左相机内参数

cameraMatrix1 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02], 
                          [0.000000e+00, 7.215377e+02, 1.728540e+02], 
                          [0.000000e+00, 0.000000e+00, 1.000000e+00,]])
#右相机内参数
cameraMatrix2 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02], 
                          [0.000000e+00, 7.215377e+02, 1.728540e+02], 
                          [0.000000e+00, 0.000000e+00, 1.000000e+00,]])

'''
cameraMatrix1 = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02], 
                          [0.000000e+00, 9.569251e+02, 2.241806e+02], 
                          [0.000000e+00, 0.000000e+00, 1.000000e+00,]])
#右相机内参数
cameraMatrix2 = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], 
                          [0.000000e+00, 9.019653e+02, 2.242509e+02], 
                          [0.000000e+00, 0.000000e+00, 1.000000e+00,]])
'''

#左右相机畸变系数:[k1, k2, p1, p2, k3]
distCoeffs1 = np.array([[-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02]])
distCoeffs2= np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

#旋转矩阵
R2 = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03 ], 
              [5.251945e-03, 9.999804e-01, -3.413835e-03 ], 
              [4.570332e-03, 3.389843e-03, 9.999838e-01]])

R3 = np.array([[9.995599e-01, 1.699522e-02, -2.431313e-02], 
              [-1.704422e-02, 9.998531e-01, -1.809756e-03], 
              [2.427880e-02, 2.223358e-03, 9.997028e-01]])

R =  np.linalg.inv(R2)*R3

#平移矩阵
T2 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])

T3 = np.array([-4.731050e-01, 5.551470e-03, -5.250882e-03])

T = T2 - T3

#矫正一张图像看看是否完成了极线矫正
start_time = time.time()
fname1 = 'E:/pics/000000_10.png'
fname2 = 'E:/pics/000000_11.png'
img1 = cv2.imread(fname1)
img2 = cv2.imread(fname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
w,h = gray1.shape[::-1]

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1) #cv2.CV_16SC2
map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

result1 = cv2.remap(img1, map1_1, map1_2, cv2.INTER_LINEAR)
result2 = cv2.remap(img2, map2_1, map2_2, cv2.INTER_LINEAR)

print("矫正处理时间%f(s)" % (time.time() - start_time))
result = np.concatenate((result1, result2), axis=1)
result[::20, :] = 0
cv2.namedWindow("rectification",0)
cv2.imshow("rectification",result)
#cv2.imwrite("E:/pics/rec_k0.png", result)

#计算视差图并显示
#视差计算，对SGBM方法进行GPU加速，后台选择GPU设备运行
@cuda.jit(device=True)
def SGBM(imgL, imgR):
    #SGBM参数设置
    blockSize = 11
    img_channels1 = img1.shape[2]
    img_channels2 = img2.shape[2]
    stereo = cv2.StereoSGBM_create(minDisparity = 0,
                                   numDisparities = 128,
                                   blockSize = blockSize,
                                   P1 = 4 * img_channels1 * blockSize * blockSize,
                                   P2 = 32 * img_channels2 * blockSize * blockSize,
                                   disp12MaxDiff = 1,
                                   preFilterCap = 63,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32,
                                   mode = cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)#除以16得到真实视差图
    return disp

disp = SGBM(result1, result2)
disp = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imwrite("E:/pics/disp_k0.png", disp)
cv2.imshow("disparity", disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
