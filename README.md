# stereo_project
关于立体视觉的项目尝试，此处仅给出编程题的代码实现和相应注释。  
运行环境：Windows10操作系统，python3.6，OpenCV4.1.2  
分为以下三个部分： 
## Part Ⅰ：Camera Basics
### 问题6：Provided a series of images taken by the camera, can you calibrate the camera with OpenCV functions? 
#### Answer
代码实现为calibration.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/calibration.py  
使用OpenCV文档中提供的left.zip中所有图片进行相机标定。  
#### Tip
请注意程序中的图片路径均为相对路径。   
#### Reference
http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/

### 问题7：Undistort the images with the calibration results you computed. 
#### Answer
代码实现为undistort.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/undistort.py  
使用OpenCV文档中提供的left.zip中left12.jpg为例进行校正。  
#### Tip
请注意程序中的图片路径均为相对路径。   
#### Reference
http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/  

### 问题8：Learn about Zhang’s method for camera calibration. Can you implement Zhang’s method? 
#### Answer
代码实现为文件夹zhang's implementation，代码地址https://github.com/Hhhana/stereo_project/tree/master/zhang's_implementation  
共有5个python文件，其中main.py是主程序用于实现张氏标定法。使用OpenCV文档中提供的left.zip中所有图片进行标定。  
#### Tip
请注意程序中的图片路径均为相对路径。   
#### Reference
https://blog.csdn.net/qq_40369926/article/details/89251296?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight

## Part Ⅱ：Binocular Basics
### 问题12：Now please use OpenCV to calibrate the binocular cameras with images in left.zip and right.zip respectively. Report the results.
### 问题14：Now use OpenCV to rectify the left and the right images with the calibration results obtained from Problem 13.
#### Answer
两个问题代码实现为rectification.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/rectification.py    
使用OpenCV文档中提供的left.zip和right.zip图片来获取双目标定参数，同时利用SGBM方法对第一张图进行双目矫正。  
#### Tip
请注意程序中的图片路径均为相对路径。   
#### Reference
https://zhuanlan.zhihu.com/p/98169184  

## Part Ⅲ： Stereo Matching

### 问题17：Can you use OpenCV to compute the disparity maps for the images you used for stereo calibration? 
#### Answer
代码实现为disparity.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/disparity.py  
使用OpenCV文档中提供的left.zip和right.zip图片来获取双目标定参数，选取第一幅图获取视差图。   
#### Tip
请注意程序中的图片路径均为相对路径。    
#### Reference
https://www.cnblogs.com/er-gou-zi/p/11926159.html  

### 问题20：Can you implement SGM with GPU (CUDA) acceleration? Test your implementation on KITTI data. Report your results (speed and error).
#### Answer
代码实现为SGBM_GPU.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/SGBM_GPU.py  
使用KITTI中标定文件calib_cam_to_cam获取双目标定参数，选取第一幅图的左右视图0000_01.png和0000_11.png获取其视差图。   
#### Tip
请注意程序中的图片路径均为相对路径。  
#### Reference
https://blog.csdn.net/qq_28023365/article/details/87970505
