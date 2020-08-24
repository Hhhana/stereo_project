# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:04:48 2020

@author: Hana Luo
"""
import numpy as np
import cv2
import os
import time

# 0.基本配置
show_corners = False

image_number = 13
board_size = (9, 6)  # 也就是boardSize
square_Size = 20

image_lists = []  # 存储获取到的图像
image_points = []  # 存储图像的点

# 1.读图,找角点
image_dir = "E:/pics"
image_names = []

[image_names.append(image_dir + "/left%02d.jpg" % i) for i in
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]]  # 没有10
[image_names.append(image_dir + "/right%02d.jpg" % i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]]
print(len(image_names))

for image_name in image_names:
    print(image_name)
    image = cv2.imread(image_name, 0)
    found, corners = cv2.findChessboardCorners(image, board_size)  # 粗查找角点
    if not found:
        print("ERROR(no corners):" + image_name)
        # 展示结果
    if show_corners:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, board_size, corners, found)
        cv2.imwrite(image_name.split(os.sep)[-1], vis)
        cv2.namedWindow("xxx", cv2.WINDOW_NORMAL)
        cv2.imshow("xxx", vis)
        cv2.waitKey()
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), term)  # 精定位角点
    image_points.append(corners.reshape(-1, 2))
    image_lists.append(image)

# 2. 构建标定板的点坐标，objectPoints
object_points = np.zeros((np.prod(board_size), 3), np.float32)
object_points[:, :2] = np.indices(board_size).T.reshape(-1, 2)
object_points *= square_Size
object_points = [object_points] * image_number

# 3. 分别得到两个相机的初始CameraMatrix
h, w = image_lists[0].shape
camera_matrix = list()

camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[:image_number], (w, h), 0))
camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[image_number:], (w, h), 0))

# 4. 双目视觉进行标定
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(object_points, image_points[:image_number], image_points[image_number:], camera_matrix[0],
        None, camera_matrix[1], None, (w, h),
        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
        criteria=term)

# 5. 矫正一张图像看看，是否完成了极线矫正
start_time = time.time()
fname1 = 'E:/Project_Stereo_left/left/left01.jpg'
fname2 = 'E:/Project_Stereo_right/right/right01.jpg'   
img1 = cv2.imread(fname1)
img2 = cv2.imread(fname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

map1_1, map1_2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1) #cv2.CV_16SC2
map2_1, map2_2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

result1 = cv2.remap(img1, map1_1, map1_2, cv2.INTER_LINEAR)
result2 = cv2.remap(img2, map2_1, map2_2, cv2.INTER_LINEAR)
print("变形处理时间%f(s)" % (time.time() - start_time))

result = np.concatenate((result1, result2), axis=1)
result[::20, :] = 0
cv2.namedWindow("rectification",0)
cv2.imshow("rectification",result)
# cv2.imwrite("E:/pics/rec13.png", result)

# 8. 计算视差图并显示
#视差计算
def SGBM(imgL, imgR):
    #SGBM参数设置
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = -1,
                                   preFilterCap = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 100,
                                   mode = cv2.STEREO_SGBM_MODE_HH)
                                   #mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                                   #mode = cv2.STEREO_SGBM_MODE_SGBM)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)#除以16得到真实视差图
    return disp

# 9.对视差图进行空洞填充
def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv # 或符号前后条件

	return im_out
  
# 测试
disp = SGBM(result1, result2)
cv2.normalize(disp,disp,0,255,cv2.NORM_MINMAX)
#ndisp = fillHole(disp)
#cv2.imwrite("E:/pics/disp01_hole2.png", disp)
cv2.imshow("disp", disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
