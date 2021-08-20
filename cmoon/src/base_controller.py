#!/usr/bin/env python
# coding: UTF-8 

import rospy
from geometry_msgs.msg import Twist


class Base:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

    def rotate(self, speed):
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = speed
        self.pub.publish(self.twist)

    def stop(self):
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = 0
        self.pub.publish(self.twist)


if __name__ == '__main__':
    try:
        rospy.init_node('base', anonymous=True)
        Base('name')
    except rospy.ROSInterruptException:
        pass
