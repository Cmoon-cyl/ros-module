#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os


class Main:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        rospy.Subscriber('/k4a/rgb/image_raw', Image, self.changeform, queue_size=1, buff_size=52428800)
        # rospy.Subscriber("/usb_cam/image_raw", Image, self.changeform,queue_size=1,buff_size=52428800)
        self.rgb = rospy.Publisher('/rgb_image', Image, queue_size=10)
        self.path = os.path.dirname(os.path.dirname(__file__))
        self.bgra_image = None
        self.rgb_image = Image()
        self.cv_bridge = CvBridge()

    def changeform(self, image):
        self.bgra_image = self.cv_bridge.imgmsg_to_cv2(image, 'rgb8')
        # cv2.imwrite(self.path + '/photo/cvbridge.jpg', self.rgb_image.data)
        self.rgb_image = self.cv_bridge.cv2_to_imgmsg(self.bgra_image, encoding="rgb8")
        self.rgb.publish(self.rgb_image)


if __name__ == '__main__':
    try:
        Main('kinect_test')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
