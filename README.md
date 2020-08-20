# stereo_project
关于立体视觉的项目尝试，此处仅给出编程题的代码实现和相应注释。
运行环境：Windows10操作系统，python3.6，OpenCV4.1.2
分为以下三个部分： 
## Part Ⅰ：Camera Basics
### 问题6：Provided a series of images taken by the camera, can you calibrate the camera with OpenCV functions? 
代码实现为calibration.py，使用OpenCV文档中提供的left.zip中所有图片进行相机标定。  
tips：运行前需要修改图片路径  
line19 'images = glob.glob('E:/Project_Stereo_left/left/*.jpg')''images = glob.glob('E:/Project_Stereo_left/left/*.jpg')'
reference：http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/

### 问题7：Undistort the images with the calibration results you computed. 
代码实现为undistort.py，使用OpenCV文档中提供的left.zip中left12.jpg为例进行校正。
tips：运行前需要修改图片路径
line19 'images = glob.glob('E:/Project_Stereo_left/left/*.jpg')'
line41 'img = cv.imread('E:/Project_Stereo_left/left/left12.jpg')'
reference：http://woshicver.com/Eighth/7_1_%E7%9B%B8%E6%9C%BA%E6%A0%A1%E5%87%86/

### 问题8：Learn about Zhang’s method for camera calibration. Can you implement Zhang’s method? 
代码实现为文件夹zhang's implementation，共有5个python文件，其中main.py是主程序用于实现张氏标定法。使用OpenCV文档中提供的left.zip中所有图片进行标定。
tips：运行前需要修改图片路径
line41 'images = glob.glob('E:/Project_Stereo_left/left/*.jpg')''images = glob.glob('E:/Project_Stereo_left/left/*.jpg')'
reference：https://blog.csdn.net/qq_40369926/article/details/89251296?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.edu_weight

## Part Ⅱ：Binocular Basics 
## Part Ⅲ： 
