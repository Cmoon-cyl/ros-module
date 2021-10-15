#!/usr/bin/env python3
# coding: UTF-8
# Created by Cmoon

import rospy
import cv2
from aip import AipBodyAnalysis, AipFace
import base64
import os
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from pyKinectAzure import pyKinectAzure, _k4a


class Detector(object):
    def __init__(self):
        self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'
        self.weights = r'/home/cmoon/workingspace/src/cmoon/src/weights/yolov5s6.pt'

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def take_photo(self, device='camera'):
        """电脑摄像头拍照保存"""
        if device == 'k4a' or device == 'kinect':
            self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'
            self.k4a = pyKinectAzure(self.modulePath)
            self.k4a.device_open()
            device_config = self.k4a.config
            device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
            print(device_config)
            self.k4a.device_start_cameras(device_config)
            path = self.photopath + '/photo.jpg'
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                if color_image_handle:
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    cv2.imwrite(path, color_image)
                    if 'photo.jpg' in os.listdir(self.photopath):
                        self.k4a.image_release(color_image_handle)
                        self.k4a.capture_release()
                        break
            self.k4a.device_stop_cameras()
            self.k4a.device_close()
        else:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.open(0)
            flag, frame = cap.read()
            path = self.photopath + '/photo.jpg'
            cv2.imwrite(path, frame)
            cap.release()
        return path

    def get_attr(self, *key):
        return

    def detect(self, attributes=None, device='camera', mode=None, *keys):
        """电脑摄像头拍照检测"""
        path = self.take_photo(device)
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
        if outcome['error_code'] == 0:
            result = {}
            # result['location'] = outcome['result']['face_list'][0]['location']
            for items in attributes:
                result[items] = outcome['result']['face_list'][0][items]

        return result


class ObjectDetector(Detector):
    def __init__(self):
        self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.half = self.device != 'cpu'
        self.imgsz = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.05
        self.classes = None
        self.list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light',
                     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                     'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                     'frisbee',
                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard',
                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                     'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                     'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear',
                     'hair drier', 'toothbrush']
        super().__init__()

    def load_model(self):
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        # stride = int(model.stride.max())  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if self.half:
            model.half()  # to FP16
        if self.device != 'cpu':
            model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        return model

    def process_img(self, stride0, img0):
        img = letterbox(img0, self.imgsz, stride=stride0, auto=True)[0]
        # img = letterbox(img0, self.imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def pred(self, model, img0):
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        img = self.process_img(stride, img0)
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, agnostic=False)
        result = []
        name = []
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(img0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    c = int(cls)
                    label = (f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    result.append(aim)
            if len(result):
                for classes in result:
                    name.append(self.list[int(classes[0])])
            img0 = annotator.result()
            return name, result

    def judge(self, mode, name=None, attributes=None):
        if mode == 'realtime':
            print(name)
            flag = cv2.waitKey(1) & 0xFF == ord('q')
        elif mode == 'find':
            flag = cv2.waitKey(1) and attributes in name
        else:
            flag = cv2.waitKey(1) and name != []
        return flag

    def detect(self, attributes=None, device='camera', mode='realtime', *keys):
        if attributes is not None:
            mode = 'find'
        model = self.load_model()
        name = []
        if device == 'k4a' or device == 'kinect':
            self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'
            self.k4a = pyKinectAzure(self.modulePath)
            self.k4a.device_open()
            device_config = self.k4a.config
            device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
            print('Kinect opened!')
            self.k4a.device_start_cameras(device_config)
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                if color_image_handle:
                    img0 = self.k4a.image_convert_to_numpy(color_image_handle)
                    name, result = self.pred(model, img0)
                    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('yolo', 1280, 720)
                    cv2.imshow('yolo', img0)
                    if self.judge(mode, name, attributes):
                        break

            self.k4a.device_stop_cameras()
            self.k4a.device_close()

        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.open(0)
            while cap.isOpened():
                flag, img0 = cap.read()
                name, result = self.pred(model, img0)
                # print(name)
                cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('yolo', 640, 640)
                cv2.imshow('yolo', img0)
                if self.judge(mode, name, attributes):
                    break

            cap.release()
        return name


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        # face = FaceDetector()
        # result1 = face.detect(attributes=['age', 'gender', 'glasses', 'beauty', 'mask'], device='cam')
        # print(result1)

        # body = BodyDetector()
        # result2 = body.detect(
        #     ['age', 'gender', 'upper_wear', 'upper_wear_texture', 'upper_wear_fg', 'upper_color',
        #      'lower_wear', 'lower_color', 'face_mask', 'glasses', 'headwear', 'bag'], device='cam')
        # print(result2)

        yolo = ObjectDetector()
        name = yolo.detect(device='k4a', mode='realtime', attributes='laptop')
        print('name:{}'.format(name))

    except rospy.ROSInterruptException:
        pass
