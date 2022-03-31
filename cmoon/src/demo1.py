#!/usr/bin/env python3
# coding: UTF-8 

import rospy
from base_controller import Base  # 底盘运动模块
from Detector import ObjectDetector
import math


class Tester:
    def __init__(self):
        self.base = Base()
        self.yolo = ObjectDetector()
        self.list = []
        self.threshold = 10  # 判断是否舍去的阈值,需要调整
        self.resolution = 640  # 相机分辨率
        self.range = 0.5  # 物体出现在画面中的范围(50%)
        self.rotate = 1.0  # 底盘旋转速度

    def run(self):
        """
        resolution是相机分辨率
        rotate是底盘旋转速度
        range是占画面中间区域百分之多少
        返回的self.result是列表,里面存的是类,有name,box,x,y四个属性
        name是检测到的物品名,box是bounding box的xyxy,x和y是中心点坐标
        """
        while not rospy.is_shutdown():
            self.result = self.yolo.detect(device='cam', mode='detect', depth=False,
                                           rotate=self.rotate, range=self.range)

            for object in self.result:
                print(object)
                if len(self.list):
                    if self.distance(object, self.list[-1]) >= self.threshold:
                        self.list.append(object)
                        print(f'Object {object.name} ACCEPTED!!!')
                    else:
                        print(f'Object {object.name} REJECTED!!!')
                else:
                    self.list.append(object)
                    print(f'Object {object.name} ACCEPTED!!!')

            if len(self.list) >= 5:
                print(f'final result:{self.list}')
                break

    def distance(self, point1, point2):
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)
        tester = Tester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
