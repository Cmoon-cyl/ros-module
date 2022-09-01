# ros-module
把cmoon这个文件夹放进工作空间的src里编译,缺啥装啥  
可能会缺 rbx1,arbotix,robot-state-publisher,pdfkit,wkhtmltopdf  
不要忘了给python文件加可执行权限  
部分语句:  
roslaunch cmoon rviz_monitu.launch 启动仿真环境  
roslaunch cmoon  yolo2ros.launch yolo实时检测 (需要azure kinect ros 驱动并且插上上摄像头)  
rosrun cmoon controller.py 可以实现语音加导航,捡垃圾任务实时检测前的所有功能  (需要先开riddle.launch)  
roslaunch cmoon turtle_graph 可以乌龟画图  

使用Detector：  
1.依赖:pip3 install baidu-aip  
 pip3 install pyk4a  
 使用ObjectDetector要在最上面Detector类的初始化函数内更改pt模型的路径  
2.导入：from Detector import FaceDetector，BodyDetector,ObjectDetector  
3.实例化：self.body=BodyDetector(),self.face=FaceDetector(),self.yolo=ObjectDetector()  
4.调用：  
人脸及身体特征检测：  
result=self.body.detect(['age','gender','glasses',upper_wear'],device='k4a')传入要检测的特征及设备。设备参数不传默认电脑摄像头，传入k4a使用kinect。摄像头拍照一次并检测，返回结果字典  
result=self.body.get_attr('/home/cmoon/photo.jpg',['age','gender'])传入图片路径，要检测的特征，返回结果字典  
face与body调用方法相同  
物体检测：
result = self.yolo.detect(device='cam', mode='realtime', find=None, depth=False, rotate=False, range=0.5)
device参数可选k4a或kinect来使用kinect相机，传cam或不传参数使用电脑摄像头.   
mode参数可选realtime,开启实时检测,按q停止,不传参数默认检测到一次物品就停止检测.  
find参数可传入能识别的一个物品名称字符串,例如'bottle'，可以一直检测到出现bottle才停止,不传则默认都检测,也可更改self.class来选择要检测哪些物品.    
depth可开启深度相机，返回物体深度值，准确度有待测试  
rotate传入转速即可控制底盘开始旋转  
range传入参数控制摄像头画面中多大范围内检测到的物体被计入  

  
重要更新：  
为了使Python3的模块能够使用，现将controller默认py版本改为3，请将controller.py首行改为#!/usr/bin/env python3  
另外，tf由于不支持python3，现取消base_controller调用tf模块，请替换base_controller.py重新编译。base模块功能依旧可以使用  
若将controller的python版本改为python3报错rospkg，运行pip3 install rospkg  

  
启动垃圾任务流程:  
新版：  
1.roslaunch cmoon rviz_monitu.launch(开启仿真)  
2.roslaunch riddle2019 riddle.launch(语音识别)  
3.rosrun cmoon controller1.py  

旧版：  
使用azure kinect:  
1.kinect.py里订阅者订阅k4a,注释掉订阅usb_cam的那句话    
2.roslaunch cmoon rviz_monitu.launch(开启仿真)  
3.roslaunch azure_kinect_ros_driver rectify_test.launch(需要连上azure kinect)  
4.roslaunch riddle2019 riddle.launch(开启语音识别)  
5.roslaunch cmoon rubbish.launch(主程序)  
  
使用笔记本摄像头:  
1.kinect.py里订阅者订阅usb_cam,注释掉订阅k4a的那句话    
2.roslaunch cmoon rviz_monitu.launch(开启仿真)  
3.rosrun usb_cam usb_cam_node(没有的sudo apt-get install ros-melolic-usb-cam)  
4.roslaunch riddle2019 riddle.launch(开启语音识别)  
5.roslaunch cmoon rubbish.launch(主程序)  

乌龟画图例程:   
1.roslaunch cmoon turtle_graph.launch  
2.在终端根据提示输入要画的图形(squ,tri_60,tri_90,rec)  
或者:  
1.rosrun turtlesim turtlesim_node  
2.rosrun cmoon turtle_graph.py squ(直接在语句后面跟上要画图形名称就行,不加的话在终端输入也行)  
转换成rviz仿真:  
1.roslaunch cmoon rviz_graph.launch  
2.在终端输入要画的图形(squ,tri_60,tri_90,rec)  

使用骨架识别:  
1.将cmoon launch里的rectify_body.launch 放入azure kinect ros driver包  
2.roslaunch azure_kinect_ros_driver rectify_body.launch  
3.roslaunch cmoon rviz_body.launch  
4.roslaunch cmoon kinect2base.launch  


