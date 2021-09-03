#!/usr/bin/env python
# coding: UTF-8 

import rospy
from visualization_msgs.msg import MarkerArray
from soundplayer import Soundplayer
from geometry_msgs.msg import PointStamped

JOINT = ['pelvis', 'navel', 'chest', 'neck', 'left clavicle', 'left shoulder', 'left elbow', 'left wrist', 'left hand',
         'left handtip', 'left thumb', 'right clavicle', 'right shoulder', 'right elbow', 'right wrist', 'right hand',
         'right handtip', 'right thumb', 'left hip', 'left knee', 'left ankle', 'left foot', 'right hip', 'right knee',
         'right ankle', 'right foot', 'head', 'nose', 'left eye', 'left ear', 'right eye', 'right ear', 'count']


class Findbody:
    def __init__(self):
        rospy.Subscriber('/k4a/body_tracking_data', MarkerArray, self.save)
        self.show_point = rospy.Publisher('/clicked_point', PointStamped, queue_size=10)
        self.soundplayer = Soundplayer()
        self.id = None
        self.px = None
        self.py = None
        self.pz = None
        self.ox = None
        self.oy = None
        self.oz = None
        self.ow = None
        self.point = PointStamped()

    def save(self, msg):
        if len(msg.markers) != 0:
            print('saving')
            self.id = msg.markers[2].id / 100
            self.joint = msg.markers[2].id % 100
            self.px = msg.markers[2].pose.position.x
            self.py = msg.markers[2].pose.position.y
            self.pz = msg.markers[2].pose.position.z
            self.ox = msg.markers[2].pose.orientation.x
            self.oy = msg.markers[2].pose.orientation.y
            self.oz = msg.markers[2].pose.orientation.z
            self.ow = msg.markers[2].pose.orientation.w
            self.point.header.frame_id = 'depth_camera_link'
            self.point.point.x = self.px
            self.point.point.y = self.py
            self.point.point.z = self.pz
            self.show_point.publish(self.point)

    def find(self):
        while not rospy.is_shutdown():
            if self.id is not None:
                print('I have found a person.')
                self.soundplayer.say('I have found a person.')
                rospy.loginfo('{},{},{},{},{}'.format(self.id, self.joint, self.px, self.py, self.pz))
                break
            else:
                print('finding')


if __name__ == '__main__':
    try:
        rospy.init_node('findbody', anonymous=True)
        finder = Findbody()
        # finder.find()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
