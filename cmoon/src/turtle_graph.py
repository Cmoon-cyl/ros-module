#!/usr/bin/env python3
# coding: utf-8
# Created by Cmoon

import math
import rospy
import sys
from turtlesim.msg import Pose  # Pose数据类型包含乌龟的坐标和角度
from geometry_msgs.msg import Twist  # Twist数据类型包含线速度和角速度


class Turtle:
    def __init__(self, name, graph):
        rospy.init_node(name)  # 初始化节点
        rospy.Subscriber('/turtle1/pose', Pose, self.control)  # 实例化订阅者，参数为订阅的话题名，消息类型，回调函数
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)  # 实例化发布者，参数为发布的话题名，消息类型，队列长度
        self._graph = graph
        self.size = 5  # 图形的大小(3-5)
        self.kp1 = 6  # 走直线的比例控制参数
        self.kp2 = 4  # 转角度的比例控制参数
        self.kd = 10  # 走直线的微分控制参数
        self.aim = 0.025  # 直走的误差值
        self.aim_rotate = 0.008  # 转弯的误差值
        self.vel_cmd = Twist()  # 实例化Twist消息类型的消息
        self.point = {}  # 储存所走路径目标点
        self.x = None  # 乌龟目前所在x坐标
        self.y = None  # 乌龟目前所在y坐标
        self.theta = None  # 乌龟目前角度
        self.goal = None  # 乌龟下一个要到达的目标点
        self.error = None  # 现在距离目标点的误差值
        self.flag = 0  # flag和lock实现走直线和转角度的互锁，执行完一个才能执行另一个
        self.lock = 0
        self.key = 0

    def control(self, pose):
        """订阅的回调函数"""
        self.choose_graph(self._graph, pose)  # 设定要走什么形状

    def choose_graph(self, graph, pose):
        """根据传入的图形设定目标点，获取当前位置，控制运动"""
        self.set_goal_points(pose, self.size)
        rospy.loginfo(self.point['squ'][0][0])
        self.get_present_point(pose)
        self.go_graph(graph, self.kp1, self.kp2, self.kd, self.aim)

    def set_goal_points(self, pose, size=5):
        """设定不同形状的目标点，坐标和角度"""
        if self.key == 0:  # 只设定一次
            self.point = {'squ': [[pose.x, pose.y, 0],
                                  [pose.x + size, pose.y, math.pi / 2],
                                  [pose.x + size, pose.y + size, math.pi],
                                  [pose.x, pose.y + size, - math.pi / 2]
                                  ],

                          'rec': [[pose.x, pose.y, pose.theta],
                                  [pose.x + 5, pose.y, pose.theta + math.pi / 2],
                                  [pose.x + 5, pose.y + 3, pose.theta + math.pi],
                                  [pose.x, pose.y + 3, pose.theta - math.pi / 2]
                                  ],

                          'tri_60': [[pose.x, pose.y, pose.theta],
                                     [pose.x + size, pose.y, math.pi * 2 / 3],
                                     [pose.x + size / 2, pose.y + (size / 2 * math.tan(math.pi / 3)),
                                      pose.theta - math.pi * 2 / 3]
                                     ],

                          'tri_90': [[pose.x, pose.y, 0],
                                     [pose.x + size, pose.y, math.pi / 2],
                                     [pose.x + size, pose.y + size, -3 * math.pi / 4],
                                     ]
                          }
            self.key += 1

    def get_present_point(self, pose):
        """获取当前坐标和角度"""
        self.x = pose.x
        self.y = pose.y
        self.theta = pose.theta

    def go_graph(self, graph, kp1=4, kp2=4, kd=15, aim=0.025):
        """控制运动"""
        self.go_line(graph, kp1, kd, aim)
        self.rotate(graph, kp2)

    def find_closest_point(self, graph):
        """寻找现在距离哪个点最近"""
        # 计算现在坐标和其他所有目标点的距离，传入distance列表
        distance = [math.sqrt((self.x - self.point[graph][i][0]) ** 2 + (self.y - self.point[graph][i][1]) ** 2) for i
                    in range(len(self.point[graph]))]
        closest_point = distance.index(min(distance))  # 获取列表中最小的点的索引
        return closest_point

    def go_line(self, graph, kp=2, kd=15, aim=0.08, ):
        """控制走直线"""
        if self.lock == 0:  # 和旋转互锁
            if self.flag == 0:  # 第一次进函数执行一次初始化
                closest_point = self.find_closest_point(graph)  # 获取最近的点
                self.goal = closest_point + 1 if closest_point != len(self.point[graph]) - 1 else 0  # 计算下一个目标点
                self.error = self.size  # 初始误差为设定的图形大小
                self.flag += 1  # 保证初始化只执行一次
            last_error = self.error  # 上一次的误差
            self.error = math.sqrt(
                (self.x - self.point[graph][self.goal][0]) ** 2 + (
                        self.y - self.point[graph][self.goal][1]) ** 2)  # 计算现在的误差
            if abs(self.error) > aim:  # 误差大于设定精度时前进
                self.vel_cmd.linear.x = kp * self.error + kd * (self.error - last_error)  # pd控制计算当前速度
                rospy.loginfo('mode:going')
                rospy.loginfo('goal:{},error:{},speed:{}'.format(self.goal, self.error, self.vel_cmd.linear.x))
            else:
                self.vel_cmd.linear.x = 0
                self.lock = 1  # 解锁旋转
            self.pub.publish(self.vel_cmd)  # 发布乌龟速度

    def rotate(self, graph, kp=2):
        """控制旋转"""
        if self.lock == 1:  # 和直走互锁
            if self.flag == 1:  # 第一次进函数执行一次初始化函数
                closest_point = self.find_closest_point(graph)  # 获取最近的点
                self.goal = closest_point + 1 if closest_point != len(self.point[graph]) - 1 else 0  # 计算下一个目标点
                self.flag += 1  # 保证只执行一次初始化
            self.error = abs(self.theta - self.point[graph][self.goal - 1][2])  # 计算当前误差
            if self.error > self.aim_rotate:  # 误差大于0.01时保持旋转
                self.vel_cmd.angular.z = kp * (self.error + 0.1)  # p控制计算旋转速度
                rospy.loginfo('mode:rotating')
                rospy.loginfo('goal:{},error:{},speed:{}'.format(self.goal, self.error, self.vel_cmd.angular.z))
            else:
                self.vel_cmd.angular.z = 0
                self.flag = 0
                self.lock = 0
            self.pub.publish(self.vel_cmd)  # 发布乌龟速度


if __name__ == '__main__':
    graph_list = ['squ', 'rec', 'tri_60', 'tri_90']
    if len(sys.argv) > 1 and sys.argv[1] in graph_list:  # 实现rosrun xx yy.py squ直接跑对应图形
        command = sys.argv[1]
    else:
        command = input('Please input graph name(squ tri_60 tri_90 rec): ')  # 没在命令输对图形名称时提示输入
    try:
        Turtle('turtle_graph', str(command))  # 实例化Turtle，传入初始化的节点名s
        rospy.spin()  # 循环监听callback
    except rospy.ROSInterruptException:
        rospy.loginfo("Keyboard interrupt.")
