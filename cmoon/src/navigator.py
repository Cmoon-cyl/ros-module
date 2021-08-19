#!/usr/bin/env python3
# !coding=utf-8
# Created by Cmoon

import rospy
from std_srvs.srv import Empty
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from soundplayer import Soundplayer

LOCATION = {
    'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]],
    'living room': [[-0.476640, -4.946882, 0.000000], [0.000000, 0.000000, 0.808888, 0.587962]],
    'kitchen': [[-1.658400, -0.046712, 0.000000], [0.000000, 0.000000, -0.986665, 0.162761]],
    'bedroom': [[3.859466, -2.201285, 0.000000], [0.000000, 0.000000, -0.247601, -0.968862]],
    'dining room': [[3.583689, 0.334696, 0.000000], [0.000000, 0.000000, -0.820933, -0.571025]],
    'garage': [[0.166213, 3.886673, 0.000000], [0.000000, 0.000000, -0.982742, 0.184983]],
}


class Navigator:
    def __init__(self):
        self.location = LOCATION
        self.goal = MoveBaseGoal()  # 实例化MoveBaseGoal这一消息类型
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        rospy.sleep(1)  # 等一秒,增加稳定性
        self.clear_costmap_client = rospy.ServiceProxy('move_base/clear_costmaps', Empty)
        rospy.sleep(1)
        rospy.loginfo('Navigation ready...')

    def goto(self, place):
        point = self.set_goal("map", self.location[place][0], self.location[place][1])  #设置导航点
        self.go_to_location(point)
        print('I have got the  ' + place)
        self.soundplayer.say('I have got the  ' + place)
        rospy.sleep(2)

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
        print('Goals set.')
        return self.goal

    def go_to_location(self, location):
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # 等待MoveBaseAction server启动
        client.wait_for_server()
        print('Ready to go.')
        while not rospy.is_shutdown():
            flag = False
            while not flag:  # 导航到指定点
                print('尝试导航...')
                self.clear_costmap_client()
                client.send_goal(location)
                client.wait_for_result()
                if client.get_state() == 3:
                    flag = True
                    break
            break


if __name__ == '__main__':
    rospy.init_node('navigation')
    Navigator()
    rospy.spin()
