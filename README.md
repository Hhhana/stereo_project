# stereo_project
关于立体视觉的项目尝试，此处仅给出编程题的代码实现和相应注释。  
运行环境：Windows10操作系统，python3.6，OpenCV4.1.2  
分为以下三个部分： 
## Part Ⅰ：Camera Basics
### 问题6：Provided a series of images taken by the camera, can you calibrate the camera with OpenCV functions? 
代码实现为calibration.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/calibration.py  
使用OpenCV文档中提供的left.zip中所有图片进行相机标定。  
tips：运行前需要修改图片路径  
line19 `images = glob.glob('E:/Project_Stereo_left/left/*.jpg')`    
reference：http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/

### 问题7：Undistort the images with the calibration results you computed. 
代码实现为undistort.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/undistort.py  
使用OpenCV文档中提供的left.zip中left12.jpg为例进行校正。  
tips：运行前需要修改图片路径  
line19 `images = glob.glob('E:/Project_Stereo_left/left/*.jpg')`  
line41 `img = cv.imread('E:/Project_Stereo_left/left/left12.jpg')`  
reference：http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/  

### 问题8：Learn about Zhang’s method for camera calibration. Can you implement Zhang’s method? 
代码实现为文件夹zhang's implementation，代码地址https://github.com/Hhhana/stereo_project/tree/master/zhang's_implementation  
共有5个python文件，其中main.py是主程序用于实现张氏标定法。使用OpenCV文档中提供的left.zip中所有图片进行标定。  
tips：运行前需要修改图片路径  
line41 `images = glob.glob('E:/Project_Stereo_left/left/*.jpg')`  
reference：https://blog.csdn.net/qq_40369926/article/details/89251296?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight

## Part Ⅱ：Binocular Basics
### 问题12：Now please use OpenCV to calibrate the binocular cameras with images in left.zip and right.zip respectively. Report the results.
### 问题14：Now use OpenCV to rectify the left and the right images with the calibration results obtained from Problem 13.
代码实现为rectification.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/rectification.py  
使用OpenCV文档中提供的left.zip和right.zip图片来获取双目标定参数，同时利用SGBM方法对第一张图进行双目矫正。 
tips：运行前需要修改图片路径  
line23 `image_dir = "E:/pics"`  
line74 `fname1 = 'E:/Project_Stereo_left/left/left01.jpg'`  
line75 `fname2 = 'E:/Project_Stereo_right/right/right01.jpg'`  
reference：https://zhuanlan.zhihu.com/p/98169184  

## Part Ⅲ： 

### 问题17：Can you use OpenCV to compute the disparity maps for the images you used for stereo calibration? 
代码实现为disparity.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/disparity.py  
使用OpenCV文档中提供的left.zip和right.zip图片来获取双目标定参数，选取第一幅图获取视差图。   
tips：运行前需要修改图片路径  
line23 `image_dir = "E:/pics"`  
line74 `fname1 = 'E:/Project_Stereo_left/left/left01.jpg'`  
line75 `fname2 = 'E:/Project_Stereo_right/right/right01.jpg'`  
reference：https://www.cnblogs.com/er-gou-zi/p/11926159.html  

### 问题20：Can you implement SGM with GPU (CUDA) acceleration? Test your implementation on KITTI data. Report your results (speed and error).
代码实现为SGBM_GPU.py，代码地址https://github.com/Hhhana/stereo_project/blob/master/SGBM_GPU.py  
使用KITTI中标定文件calib_cam_to_cam获取双目标定参数，选取第一幅图的左右视图0000_01.png和0000_11.png获取其视差图。   
tips：运行前需要修改图片路径   
line71 `fname1 = fname1 = 'E:/pics/000000_10.png'`    
line72 `fname2 = fname2 = 'E:/pics/000000_11.png'`
