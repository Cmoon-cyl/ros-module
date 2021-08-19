#!/usr/bin/env python
# coding: UTF-8 

import rospy
from navigator import Navigator
from soundplayer import Soundplayer
from voice_recognizer import Recognizer
from pdfmaker import Pdfmaker

from std_msgs.msg import String

LOCATION = {
    'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]],
    'living room': [[-0.476640, -4.946882, 0.000000], [0.000000, 0.000000, 0.808888, 0.587962]],
    'kitchen': [[-1.658400, -0.046712, 0.000000], [0.000000, 0.000000, -0.986665, 0.162761]],
    'bedroom': [[3.859466, -2.201285, 0.000000], [0.000000, 0.000000, -0.247601, -0.968862]],
    'dining room': [[3.583689, 0.334696, 0.000000], [0.000000, 0.000000, -0.820933, -0.571025]],
    'garage': [[0.166213, 3.886673, 0.000000], [0.000000, 0.000000, -0.982742, 0.184983]],
}


class Controller:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        rospy.Subscriber('/start_signal', String, self.go_point)
        self.navigator = Navigator()
        self.soundplayer = Soundplayer()
        self.recognizer = Recognizer()
        self.pdfmaker = Pdfmaker()
        self.soundplayer.play("I'm ready, please give me the commend.")
        rospy.sleep(3)
        self.recognizer.get_cmd()

    def go_point(self, place):
        self.navigator.goto(place.data)


if __name__ == '__main__':
    try:
        Controller('controller')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
