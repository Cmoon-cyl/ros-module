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
from utils.torch_utils import time_sync
from pyKinectAzure import pyKinectAzure, _k4a
# from cmoon_msgs.msg import Point
from base_controller import Base


class Detector(object):
    def __init__(self):
        self.weights = os.path.dirname(__file__) + '/weights/' + 'yolov5s6.pt'
        #self.weights = '/home/cmoon/yolov5/weights/' + 'yolov5s6.pt'

        self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'
        self.base = Base()

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
        self.t1 = 0.0
        self.t2 = 0.0
        self.K = np.array([614.86962890625, 0.0, 635.5834350585938,
                           0.0, 614.7677612304688, 364.99200439453125,
                           0.0, 0.0, 1.0]).reshape(3, 3)
        self.K = np.linalg.inv(self.K)
        self.conf_thres = 0.4
        self.iou_thres = 0.05
        #self.classes = [39, 41, 64, 67]
        self.classes = None
        self.list = None
        # self.show = rospy.Publisher('/yolo_result', Point, queue_size=10)
        # self.point = Point()
        super().__init__()

    def load_model(self):
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        # stride = int(model.stride.max())  # model stride
        self.list = model.module.names if hasattr(model, 'module') else model.names  # get class names
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
        self.t1 = time_sync()
        return img

    def pred(self, model, img0):
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        img = self.process_img(stride, img0)
        pred = model(img, augment=False, visualize=False)[0]
        self.t2 = time_sync()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, agnostic=False)
        result_ = []
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
                    # coorindate_cls.append(
                    # [names[int(cls)], xyxy[:][0].item(), xyxy[:][1].item(), xyxy[:][2].item(), xyxy[:][3].item()])
                    line = (cls, *xywh)  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    c = int(cls)
                    label = (f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    result.append([xyxy[:][0].item(), xyxy[:][1].item(), xyxy[:][2].item(), xyxy[:][3].item()])
                    result_.append(aim)
            if len(result_):
                for classes in result_:
                    name.append(self.list[int(classes[0])])
                    classes[0] = self.list[int(classes[0])]
            img0 = annotator.result()
            print('time:{:.3f}'.format(self.t2 - self.t1))
            return name, result

    # def transform_(self, n_coorindate_cls, depth_to_color_image):
    #     target_coorindate = list()  # 存放目标点三维坐标 二维列表
    #     for name, coorindate_cls in n_coorindate_cls:
    #         # print("coor:",coorindate_cls)
    #         x1, y1, x2, y2 = coorindate_cls[0], coorindate_cls[1], coorindate_cls[2], coorindate_cls[3]
    #         center_x = round((x2 - x1) / 2 + x1)
    #         center_y = round((y2 - y1) / 2 + y1)
    #         # target_coorindate = list()  # 存放目标点三维坐标 二维列表
    #         # print("coor:%s,%s" % (center_x, center_y))
    #         dep = depth_to_color_image[center_y][center_x].item()
    #         if (dep >= 100):  # 大于10cmv
    #             # image_coorindate = np.array([center_x, center_y , dep // 2]).reshape(3, 1)  # 图像坐标
    #             image_coorindate = np.array([center_x * dep, center_y * dep, dep]).reshape(3, 1)  # 图像坐标
    #             camera_coorindate = np.dot(self.K, image_coorindate)  # 相机坐标
    #             target_coorindate.append([name, camera_coorindate[0].item() / 1000,
    #                                       camera_coorindate[1].item() / 1000,
    #                                       camera_coorindate[2].item() / 1000,
    #                                       ])
    #             self.point.result = name
    #             self.point.x = camera_coorindate[0].item() / 1000
    #             self.point.y = camera_coorindate[1].item() / 1000
    #             self.point.z = camera_coorindate[2].item() / 1000
    #             self.show.publish(self.point)

    # return target_coorindate

    def xyxy2mid(self, xyxy):
        center = []
        for point in xyxy:
            # print(f'xyxy:{point}')
            center.append([(point[0] + point[2]) / 2, (point[1] + point[3]) / 2])
        return center

    def judge_range(self, x, resolution=640, range=0.5):
        left = resolution * 0.5 * (1 - range)
        right = resolution * 0.5 * (1 + range)
        return left <= x <= right

    def judge(self, mode, name=None, find=None, center=None, resolution=640, range=0.5):
        x = center[0][0] if center else None
        if mode == 'realtime':
            flag = cv2.waitKey(1) & 0xFF == ord('q')
        elif mode == 'find':
            flag = cv2.waitKey(1) and find in name and self.judge_range(x, resolution, range)
        else:
            flag = cv2.waitKey(1) and name != [] and self.judge_range(x, resolution, range)
        return flag

    def detect(self, find=None, device='camera', mode='realtime', depth=False, rotate=False, save=True,
               range=0.5, *keys):
        if find is not None:
            mode = 'find'
        model = self.load_model()
        name = []
        result = []
        output = []
        if device == 'k4a' or device == 'kinect':
            self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'
            self.k4a = pyKinectAzure(self.modulePath)
            self.k4a.device_open()
            device_config = self.k4a.config
            device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
            print('Kinect opened!')
            self.k4a.device_start_cameras(device_config)
            while True:
                if rotate:
                    self.base.rotate(rotate)
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                depth_image_handle = self.k4a.capture_get_depth_image()
                resolution = self.k4a.image_get_width_pixels(color_image_handle)

                if color_image_handle:
                    img0 = self.k4a.image_convert_to_numpy(color_image_handle)
                    name, result = self.pred(model, img0)
                    center = self.xyxy2mid(result)
                    # print(result)
                    cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('yolo', 1280, 720)
                    cv2.line(img0, (int(resolution * 0.5 * (1 - range)), 0), (int(resolution * 0.5 * (1 - range)), 480),
                             (0, 255, 0),
                             2, 4)
                    cv2.line(img0, (int(resolution * 0.5 * (1 + range)), 0), (int(resolution * 0.5 * (1 + range)), 480),
                             (0, 255, 0),
                             2, 4)
                    for point in center:
                        cv2.circle(img0, (int(point[0]), int(point[1])), 1, (0, 0, 255), 8)
                    cv2.imshow('yolo', img0)
                    if self.judge(mode, name, find, center, resolution, range):
                        if save:
                            cv2.imwrite(self.photopath + '/result.jpg', img0)
                        for item in zip(name, result, center):
                            if self.judge_range(item[2][0], resolution, range):
                                point = YoloResult(item[0], item[1], item[2][0], item[2][1])
                                # print(point)
                                output.append(point)
                        break
                    if depth_image_handle and name != list() and depth is True:
                        depth_color_image = self.k4a.transform_depth_to_color(depth_image_handle, color_image_handle)

                        # result = self.transform_(zip(name, result), depth_color_image)

                        # print("result:", result)
                # print("result:", result)

            if rotate:
                self.base.stop()

            self.k4a.device_stop_cameras()
            self.k4a.device_close()

        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.open(0)
            resolution = 640
            # rate = rospy.Rate(2000)
            while cap.isOpened():
                if rotate:
                    # print('rotating')
                    self.base.rotate(rotate)
                    # rate.sleep()
                flag, img0 = cap.read()
                name, result = self.pred(model, img0)
                center = self.xyxy2mid(result)
                cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('yolo', 640, 640)
                cv2.line(img0, (int(resolution * 0.5 * (1 - range)), 0), (int(resolution * 0.5 * (1 - range)), 480),
                         (0, 255, 0), 2, 4)
                cv2.line(img0, (int(resolution * 0.5 * (1 + range)), 0), (int(resolution * 0.5 * (1 + range)), 480),
                         (0, 255, 0), 2, 4)
                for point in center:
                    cv2.circle(img0, (int(point[0]), int(point[1])), 1, (0, 0, 255), 8)
                cv2.imshow('yolo', img0)
                if self.judge(mode, name, find, center, resolution, range):
                    if save:
                        cv2.imwrite(self.photopath + '/result.jpg', img0)
                    for item in zip(name, result, center):
                        if self.judge_range(item[2][0], resolution, range):
                            point = YoloResult(item[0], item[1], item[2][0], item[2][1])
                            # print(point)
                            output.append(point)
                    break
                # print("result:", result)
            if rotate:
                self.base.stop()
            cap.release()
        return output


class YoloResult:
    def __init__(self, name, box, x, y):
        self.name = name
        self.box = box
        self.x = x
        self.y = y
        self.distance = None

    def __str__(self):
        return f'name:{self.name},box:{self.box},x:{self.x},y:{self.y}'


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        # face = FaceDetector()
        # result1 = face.detect(attributes=['age', 'gender', 'glasses', 'beauty', 'mask'], device='cam')
        # print(result1)
        #
        # body = BodyDetector()
        # result2 = body.detect(
        #     ['age', 'gender', 'upper_wear', 'upper_wear_texture', 'upper_wear_fg', 'upper_color',
        #      'lower_wear', 'lower_color', 'face_mask', 'glasses', 'headwear', 'bag'], device='cam')
        # print(result2)
        #
        # yolo = ObjectDetector()
        # name, result = yolo.detect(device='cam', mode='realtime', find=None, depth=False, rotate=True)
        # print(name[0])
        # print(result)
        # name = yolo.detect(device='k4a', mode='realtime', attributes=None, depth=True)
        # print('name:{}'.format(name))

    except rospy.ROSInterruptException:
        pass
