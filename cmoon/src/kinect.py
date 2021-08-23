#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Main:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        # rospy.Subscriber('/k4a/rgb/image_raw', Image, self.changeform, queue_size=1, buff_size=52428800)#使用kinect实时检测
        rospy.Subscriber("/usb_cam/image_raw", Image, self.changeform, queue_size=1, buff_size=52428800)  # 使用电脑摄像头实时检测
        self.rgb = rospy.Publisher('/rgb_image', Image, queue_size=10)
        self.bgra_image = None
        self.rgb_image = Image()
        self.cv_bridge = CvBridge()

    def changeform(self, image):
        self.bgra_image = self.cv_bridge.imgmsg_to_cv2(image, 'rgb8')  # 使用cv_bridge格式转换,bgra to rgb8
        self.rgb_image = self.cv_bridge.cv2_to_imgmsg(self.bgra_image, encoding="rgb8")  # 生成图像信息
        self.rgb.publish(self.rgb_image)


if __name__ == '__main__':
    try:
        Main('kinect_test')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
