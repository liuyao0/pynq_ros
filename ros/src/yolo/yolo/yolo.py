import rclpy                            # ROS2 Python接口库
from rclpy.node import Node             # ROS2 节点类
from sensor_msgs.msg import Image       # 图像消息类型
from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
import cv2                              # Opencv图像处理库
from pynq_dpu import DpuOverlay
import sys
sys.path.append("/home/ubuntu/ros/src/yolo/yolo")
sys.path.append("/usr/local/share/pynq-venv/lib/python3.10/site-packages")
import numpy as np
from utils import pre_process
from yolo_evaluate import post_process
import torch
import time
"""
创建一个订阅者节点
"""
class Signal_Detection(Node):
    def __init__(self, name):
        super().__init__(name)                                # ROS2节点父类初始化
        self.overlay = DpuOverlay("dpu.bit")
        print("Finish read dpu.bit")
        
        self.overlay.load_model("/home/ubuntu/yolov5s.xmodel")
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
        im0 = img.copy()
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
            im0 = cv2.rectangle(im0, (int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0),2)
            print("[%d] x1:%d y1:%d x2:%d y2:%d" % (
                int(cls), 
                int(xyxy[0]), 
                int(xyxy[1]), 
                int(xyxy[2]), 
                int(xyxy[3])
            ))
        cv2.imshow("object", im0)
        cv2.waitKey(5)
       
        
    def listener_callback(self, data):
        t1 = time.time()
        self.get_logger().info('Receiving video frame')     # 输出日志信息，提示已进入回调函数
        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')  # 将ROS的图像消息转化成OpenCV图像
        self.predict(image)                           # 苹果检测
        t2 = time.time()
        print((t2-t1)*1000)
        


def main(args=None):                            # ROS2节点主入口main函数
    rclpy.init(args=args)                       # ROS2 Python接口初始化
    node = Signal_Detection("signal_detection")   # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                            # 循环等待ROS2退出
    node.destroy_node()                         # 销毁节点对象
    rclpy.shutdown()                            # 关闭ROS2 Python接口
main()