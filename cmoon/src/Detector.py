#!/usr/bin/python3
# coding: UTF-8
# Created by Cmoon

import rospy
import cv2
from aip import AipBodyAnalysis, AipFace
import base64
import os


class Detector(object):
    def __init__(self):
        self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def take_photo(self):
        """电脑摄像头拍照保存"""
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        cap.open(0)
        flag, frame = cap.read()
        path = self.photopath + '/photo.jpg'
        cv2.imwrite(path, frame)
        cap.release()
        return path

    def get_attr(self, *key):
        return

    def detect(self, attributes=None, *keys):
        """电脑摄像头拍照检测"""
        path = self.take_photo()
        print(path)
        result = self.get_attr(path, attributes)
        return result


class BodyDetector(Detector):
    """https://ai.baidu.com/ai-doc/BODY/6k3cpymz1"""

    def __init__(self):
        self.app_id = '24949761'
        self.api_key = '02d2yaps6uEDgOGRzMqs9Ggo'
        self.secret_key = 'NBDdIGGZtrLvCoi6aMxZ1Qd2uELcLiGU'
        self.client = AipBodyAnalysis(self.app_id, self.api_key, self.secret_key)
        super(BodyDetector, self).__init__()

    def get_attr(self, path=None, attributes=None):
        """
        example:path=r'/home/cmoon/图片/photo.jpeg'
        attributes=['age','gender','glasses',...]
        return：result字典,用result[age]等获得结果
        """
        attr = ''
        for items in attributes:
            attr = attr + items + ','
        image = self.get_file_content(path)
        self.client.bodyAttr(image)
        options = {}
        # options["type"] = "gender,age,glasses,upper_color,upper_wear"
        options["type"] = attr
        """ 带参数调用人体检测与属性识别 """
        outcome = self.client.bodyAttr(image, options)
        result = {}
        for items in attributes:
            result[items] = outcome['person_info'][0]['attributes'][items]['name']
        return result


class FaceDetector(Detector):
    """https://ai.baidu.com/ai-doc/FACE/ek37c1qiz"""

    def __init__(self):
        self.app_id = '24950812'
        self.api_key = 'F3431IOfGwWIItzAvXfAMUpR'
        self.secret_key = 'vLGzRyiQpAtPphgMDlE2vAdjixFsOoAE'
        self.client = AipFace(self.app_id, self.api_key, self.secret_key)
        super(FaceDetector, self).__init__()

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            base64_data = base64.b64encode(fp.read())  # 使用base64进行加密
            return str(base64_data, 'utf-8')

    def get_attr(self, path=None, attributes=None):
        """
        example:path=r'/home/cmoon/图片/photo.jpeg'
        attributes=['age','gender','glasses',...]
        return：result字典,用result[age]等获得结果
        """
        attr = ''
        for items in attributes:
            attr = attr + items + ','

        image = self.get_file_content(path)
        imageType = "BASE64"

        """ 调用人脸检测 """
        self.client.detect(image, imageType)

        """ 如果有可选参数 """
        options = {}
        options["face_field"] = attr
        options["max_face_num"] = 2
        options["face_type"] = "LIVE"
        options["liveness_control"] = "LOW"

        """ 带参数调用人脸检测 """
        outcome = self.client.detect(image, imageType, options)
        result = {}
        result['location'] = outcome['result']['face_list'][0]['location']
        for items in attributes:
            result[items] = outcome['result']['face_list'][0][items]

        return result


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        face = FaceDetector()
        result1 = face.detect(attributes=['age', 'gender', 'glasses'])

        body = BodyDetector()
        result2 = body.detect(
            ['age', 'gender', 'upper_wear', 'upper_wear_texture', 'upper_wear_fg', 'upper_color',
             'lower_wear', 'lower_color', 'face_mask', 'glasses', 'headwear', 'bag'])

        print(result1)
        print(result2)

        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
