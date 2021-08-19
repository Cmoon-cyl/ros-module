#!/usr/bin/env python
# coding: UTF-8 

import rospy
from std_msgs.msg import String
from soundplayer import Soundplayer
from pdfmaker import Pdfmaker

LOCATION = {
    'door': ['door', 'dog'],
    'kitchen': ['kitchen', 'page', 'station', 'location'],
    'living room': ['living', 'leaving', 'livingroom', 'living room'],
    'bedroom': ['bedroom', 'bad', 'bed room', 'bad room', 'bed', 'bathroom', 'beijing'],
    'hallway': ['hallway', 'whole way'],
    'dining room': ['dining', 'dying'],
    'garage': ['garage']
}


class Recognizer:
    def __init__(self):
        rospy.Subscriber('/xfspeech', String, self.talkback)
        self.wakeup = rospy.Publisher('/xfwakeup', String, queue_size=10)
        self.start_signal = rospy.Publisher('/start_signal', String, queue_size=10)
        self.cmd = None
        self.location = LOCATION
        self.goal = ''
        self._soundplayer = Soundplayer()
        self._pdfmaker = Pdfmaker()
        self.status = 0
        self.key = 1

    def talkback(self, msg):
        if self.key == 1:
            print("\n讯飞读入的信息为: " + msg.data)
            self.cmd = self.processed_cmd(msg.data)
            self.judge()

    def judge(self):
        if self.status == 0:

            response = self.analyze()

            if response == 'Do you need me ':
                self._soundplayer.play("Please say the command again. ")
                rospy.sleep(2)
                self.get_cmd()
            else:
                self.status = 1
                print(response)
                self._soundplayer.play(response)
                rospy.sleep(5)
                self._soundplayer.play("please say yes or no.")
                print('Please say yes or no.')
                rospy.sleep(2)
                self.get_cmd()

        elif ('Yes.' in self.cmd) or ('yes' in self.cmd) and (self.status == 1):

            self._soundplayer.play('Ok, I will.')
            self._pdfmaker.write('Cmd: Do you need me go to the ' + self.goal + ' and clean the rubbish there?')
            self._pdfmaker.write('Respond: Ok,I will.')
            print('Ok, I will.')
            self.start_signal.publish(self.goal)
            self.key = 0
            self.status = 0
            self.goal = ''


        elif ('No.' in self.cmd) or ('no' in self.cmd) or ('oh' in self.cmd) or ('know' in self.cmd) and (
                self.status == 1):
            self._soundplayer.play("Please say the command again. ")
            print("Please say the command again. ")
            rospy.sleep(2)
            self.status = 0
            self.goal = ''
            self.get_cmd()

    def processed_cmd(self, cmd):
        cmd = cmd.lower()
        for i in " ,.;?":
            cmd = cmd.replace(i, ' ')
        return cmd

    def get_cmd(self):
        self._soundplayer.play('Speak.')
        self.wakeup.publish('ok')

    def analyze(self):
        response = 'Do you need me'
        for (key, val) in self.location.items():
            for word in val:
                if word in self.cmd:
                    self.goal = key
                    response = response + ' go to the ' + key + ' and throw the rubbish there?'
                    break
        return response


if __name__ == '__main__':
    try:
        rospy.init_node('voice_recognition')
        Recognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
