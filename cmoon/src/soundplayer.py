#!/usr/bin/env python3
# coding: UTF-8 
# Created by Cmoon

import rospy
from sound_play.libsoundplay import SoundClient  # soundplay功能包


class Soundplayer:
    def __init__(self):
        self.soundhandle = SoundClient()  # 初始化
        rospy.loginfo('Ready to play sound...')

    def say(self, string, delay=2):
        """默认延迟2秒等话说完,可传入参数自定义等几秒"""
        rospy.sleep(1)
        self.soundhandle.stopAll()  # 停止其他在说的话,保证一次说一句
        self.soundhandle.say(string)  # 说传入的字符串
        rospy.sleep(delay)

    def play(self, string):
        """0延迟模式"""
        rospy.sleep(1)
        self.soundhandle.stopAll()  # 停止其他在说的话,保证一次说一句
        self.soundhandle.say(string)  # 说传入的字符串


if __name__ == '__main__':
    try:
        rospy.init_node('soundplayer')
        Soundplayer()
    except rospy.ROSInterruptException:
        pass
