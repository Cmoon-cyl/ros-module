#!/usr/bin/env python3
# !coding=utf-8
# Created by Cmoon

import rospy
from std_srvs.srv import Empty
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from soundplayer import Soundplayer
from base_controller import Base


class Navigator:
    def __init__(self, location):
        """
        location是字典,键是地点名字(String),值是坐标列表
        例:'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]]
        """
        self.location = location
        self.goal = MoveBaseGoal()  # 实例化MoveBaseGoal这一消息类型
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        self.base = Base()  # 我的底盘控制模块
        rospy.sleep(1)  # 等一秒,增加稳定性
        self.clear_costmap_client = rospy.ServiceProxy('move_base/clear_costmaps', Empty)  # 定义清理代价地图服务
        rospy.sleep(1)
        rospy.loginfo('Navigation ready...')

    def goto(self, place):
        """调用传入地点可直接导航"""
        point = self.set_goal("map", self.location[place][0], self.location[place][1])  # 设置导航点
        self.go_to_location(point)
        print('I have got the ' + place)
        self.soundplayer.say('I have got the ' + place)

    def go_near(self, name, position):
        """配合深度相机获取坐标可靠近物体"""
        orientation = self.base.orientation
        now_pose = self.base.position
        print(now_pose)
        pose = []
        pose.append((position[0] + now_pose[0]) / 2)
        pose.append((position[1] + now_pose[1]) / 2)
        pose.append((position[2] + now_pose[2]) / 2)
        point = self.set_goal('map', pose, orientation)
        self.go_to_location(point)
        print('I am near the ' + name)
        self.soundplayer.say('I am near the ' + name)

    def set_goal(self, name, position, orientation):
        """设置导航目标点的坐标和四元数"""
        self.goal.target_pose.header.frame_id = name
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = position[2]
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        print('Goal set.')
        return self.goal

    def go_to_location(self, location):
        """导航实现"""
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # 等待MoveBaseAction server启动
        self.client.wait_for_server()
        print('Ready to go.')
        while not rospy.is_shutdown():
            flag = False
            while not flag:  # 导航到指定点
                print('尝试导航...')
                self.clear_costmap_client()
                self.client.send_goal(location)
                self.client.wait_for_result()
                if self.client.get_state() == 3:
                    flag = True
                    break
                # 第二种写法:
                # if self.client.send_goal_and_wait(location) == 3:
                #     flag = True
                #     break

            break

    def stop(self):
        self.client.cancel_all_goals()


if __name__ == '__main__':
    rospy.init_node('navigation')
    Navigator('location')
    rospy.spin()
