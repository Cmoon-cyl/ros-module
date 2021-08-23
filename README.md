# ros-module
放进src里编译,缺啥装啥  
可能会缺 rbx1,arbotix,robot-state-publisher,pdfkit,wkhtmltopdf  
不要忘了给python文件加可执行权限  
部分语句:  
roslaunch cmoon rviz_monitu.launch 启动仿真环境  
roslaunch cmoon  yolo2ros.launch yolo实时检测 (需要azure kinect ros 驱动并且插上上摄像头)  
rosrun cmoon controller.py 可以实现语音加导航,捡垃圾任务实时检测前的所有功能  (需要先开riddle.launch)
roslaunch cmoon turtle_graph 可以乌龟画图  
  
启动垃圾任务流程:  
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

