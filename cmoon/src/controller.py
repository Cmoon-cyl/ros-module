#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import rospy
from navigator import Navigator  # 导航模块
from soundplayer import Soundplayer  # 语音合成模块
from voice_recognizer import Recognizer  # 语音识别模块和分析模块
from pdfmaker import Pdfmaker  # pdf制作模块
from std_msgs.msg import String  # String类型消息,从String.data中可获得信息

LOCATION = {  # 储存导航路径点
    'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]],
    'living room': [[-0.476640, -4.946882, 0.000000], [0.000000, 0.000000, 0.808888, 0.587962]],
    'kitchen': [[-1.658400, -0.046712, 0.000000], [0.000000, 0.000000, -0.986665, 0.162761]],
    'bedroom': [[3.859466, -2.201285, 0.000000], [0.000000, 0.000000, -0.247601, -0.968862]],
    'dining room': [[3.583689, 0.334696, 0.000000], [0.000000, 0.000000, -0.820933, -0.571025]],
    'garage': [[0.166213, 3.886673, 0.000000], [0.000000, 0.000000, -0.982742, 0.184983]],
}


class Controller:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)  # 初始化ros节点
        rospy.Subscriber('/start_signal', String, self.go_point)  # 创建订阅者订阅recognizer发出的地点作为启动信号
        self.navigator = Navigator()  # 实例化导航模块
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        self.recognizer = Recognizer()  # 实例化语音识别和逻辑判断模块
        self.pdfmaker = Pdfmaker()  # 实例化pdf导出模块
        self.soundplayer.play("I'm ready, please give me the commend.")  # 语音合成模块调用play方法传入字符串即可播放
        rospy.sleep(3)  # 睡3秒等待上面的话讲完
        self.recognizer.get_cmd()  # 获取一次语音命令

    def go_point(self, place):  # 订阅者的回调函数,传入的place是String类型消息 .data可以获取传来的信息
        self.navigator.goto(place.data)  # 导航模块调用goto方法,传入去的地点名字符串即可导航区指定地点


if __name__ == '__main__':
    try:
        Controller('controller')  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题
    except rospy.ROSInterruptException:
        pass
