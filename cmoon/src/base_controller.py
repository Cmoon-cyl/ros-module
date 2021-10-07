#!/usr/bin/env python
# coding: UTF-8 

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


class Base:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.now_pose)
        self.twist = Twist()
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.ow = 0.0
        self.position = []  # 坐标
        self.quaternion = []  # 四元数
        self.angle = 0  # 角度
        rospy.sleep(1)  # 等待获取现在位置的回调函数开始工作

    def now_pose(self, pose):
        """实时更新现在的坐标和四元数和角度"""
        self.px = pose.pose.pose.position.x
        self.py = pose.pose.pose.position.y
        self.pz = pose.pose.pose.position.z
        self.ox = pose.pose.pose.orientation.x
        self.oy = pose.pose.pose.orientation.y
        self.oz = pose.pose.pose.orientation.z
        self.ow = pose.pose.pose.orientation.w
        self.position = [self.px, self.py, self.pz]
        self.quaternion = [self.ox, self.oy, self.oz, self.ow]
        self.get_angle()

    def get_pose(self):
        """可调用获取现在的坐标和四元数"""
        print('[[{},{},{}],[{},{},{},{}]]'.format(self.px, self.py, self.pz, self.ox, self.oy, self.oz, self.ow))
        return [[self.px, self.py, self.pz], [self.ox, self.oy, self.oz, self.ow]]

    def get_angle(self):
        """可调用获取现在的角度"""
        eular = self.quad2euler(self.ox, self.oy, self.oz, self.ow)
        self.angle = eular
        return eular

    def turn(self, angle, kp=1.5, kd=0.5):
        """传入转的角度,单位°,正值左转,负值右转,范围0~180°"""
        print('turing')
        angle_radian = (float(angle) / 180) * math.pi
        start_angle = self.get_angle()
        end_angle = start_angle + angle_radian
        if end_angle > math.pi:
            end_angle = end_angle - 2 * math.pi
        elif end_angle < -math.pi:
            end_angle = end_angle + 2 * math.pi
        error = angle_radian
        rate = rospy.Rate(1000)
        print(angle_radian)
        print(self.angle)
        print(start_angle)
        print(end_angle)
        print(self.angle - end_angle)
        while abs(self.angle - end_angle) > 0.02:
            last_error = error
            error = abs(self.angle - end_angle)
            if error > math.pi:
                error = 2 * math.pi - error
            if angle >= 0:
                speed = kp * (error + 0.01)
            else:
                speed = -kp * (error + 0.01)
            print('{},{}'.format(error, speed))
            self.rotate(speed)
            rate.sleep()
        self.stop()

    def rotate(self, speed):
        """旋转"""
        # print('rotating')
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = speed
        self.pub.publish(self.twist)

    def stop(self):
        print('stop')
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = 0
        self.pub.publish(self.twist)

    def quad2euler(self, x, y, z, w):
        X = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        Y = math.asin(2 * (w * y - x * z))
        Z = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
        return Z


if __name__ == '__main__':
    try:
        rospy.init_node('base', anonymous=True)
        base = Base()
        # while not rospy.is_shutdown():
        #     base.get_angle()
        #     base.rotate(1.0)

        for i in range(10):
            degree = input('Input degree:')
            base.turn(float(degree))
            print(i)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
