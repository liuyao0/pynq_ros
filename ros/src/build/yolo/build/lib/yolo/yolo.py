import rclpy                            # ROS2 Python接口库
from rclpy.node import Node             # ROS2 节点类
from sensor_msgs.msg import Image       # 图像消息类型
from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
import cv2                              # Opencv图像处理库
from pynq_dpu import DpuOverlay
import numpy as np
import colorsys
import random
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from utils import pre_process
from yolo_evaluate import post_process
"""
创建一个订阅者节点
"""
class Signal_Detection(Node):
    def __init__(self, name):
        super().__init__(name)                                # ROS2节点父类初始化
        self.overlay = DpuOverlay("dpu.bit")
        print("Finish read dpu.bit")
        
        self.overlay.load_model("/home/ubuntu/py/yolov5.xmodel")
        print("Finish load model .")
        
        self.dpu = self.overlay.runner
        
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        
        self.shapeIn = tuple(inputTensors[0].dims)
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        
        shapeOut0 = (tuple(outputTensors[0].dims))
        shapeOut1 = (tuple(outputTensors[1].dims))
        shapeOut2 = (tuple(outputTensors[2].dims))
        self.output_data = [np.empty(shapeOut0, dtype=np.int8, order="C"), 
            np.empty(shapeOut1, dtype=np.int8, order="C"),
            np.empty(shapeOut2, dtype=np.int8, order="C")]
        
        self.sub = self.create_subscription(Image, "image_raw", self.listener_callback, 10)
        self.cv_bridge = CvBridge()
    
    def predict(self, img):
        image_data = np.array(pre_process(img, (640, 640)), dtype=np.float32)
        image = self.input_data[0]
        image[0,...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        
        output = [
            torch.tensor(self.output_data[0]).float() / 4.,
            torch.tensor(self.output_data[1]).float() / 8.,
            torch.tensor(self.output_data[2]).float() / 4.,
        ]
        for i in range(3):
            output[i] = output[i].permute(0,3,1,2)
        res = post_process(output)[0]
        res[:, :4] = res[:, :4].round()
        for *xyxy, conf, cls in reversed(res):
            cv2.rectangle(self.im0, (int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0),1)
        cv2.imshow("object", self.im0)
        
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')     # 输出日志信息，提示已进入回调函数
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')  # 将ROS的图像消息转化成OpenCV图像
        self.im0 = image.copy()
        self.predict(image)                           # 苹果检测


def main(args=None):                            # ROS2节点主入口main函数
    rclpy.init(args=args)                       # ROS2 Python接口初始化
    node = Signal_Detection("signal_detection")   # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                            # 循环等待ROS2退出
    node.destroy_node()                         # 销毁节点对象
    rclpy.shutdown()                            # 关闭ROS2 Python接口
main()