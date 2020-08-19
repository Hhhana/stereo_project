# stereo_project
关于立体视觉的项目尝试，此处仅给出编程题的代码实现和相应注释。
运行环境：Windows10操作系统，python3.6，OpenCV4.1.2
分为以下三个部分： 
## Part Ⅰ：Camera Basics
### 问题6：Provided a series of images taken by the camera, can you calibrate the camera with OpenCV functions? 
代码实现为calibration.py，使用OpenCV文档中提供的left.zip中所有图片进行相机标定。

### 问题7：Undistort the images with the calibration results you computed. 
代码实现为undistort.py，使用OpenCV文档中提供的left.zip中left12.jpg为例进行校正。

### 问题8：Learn about Zhang’s method for camera calibration. Can you implement Zhang’s method? 
代码实现为文件夹zhang's implementation，共有5个python文件，其中main.py是主程序用于实现张氏标定法。使用OpenCV文档中提供的left.zip中所有图片进行标定。


## Part Ⅱ：Binocular Basics 
## Part Ⅲ： 
