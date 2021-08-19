#!/usr/bin/env python3
# coding: UTF-8 

import rospy
from sound_play.libsoundplay import SoundClient


class Soundplayer:
    def __init__(self):
        self.soundhandle = SoundClient()
        rospy.loginfo('Ready to play sound...')

    def play(self, string):
        rospy.sleep(1)
        self.soundhandle.stopAll()
        self.soundhandle.say(string)


if __name__ == '__main__':
    try:
        rospy.init_node('soundplayer')
        Soundplayer()
    except rospy.ROSInterruptException:
        pass
