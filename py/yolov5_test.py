from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
import cv2                              # Opencv图像处理库
from pynq_dpu import DpuOverlay
import numpy as np
import pkg_resources as pkg
import torch
import torchvision
import time
import glob
from yolo_evaluate import post_process, raw_post_process
print('Finish import .')
class_name = ['pl5',
'pl10',
'pl15',
'pl20',
'pl30',
'pl35',
'pl40',
'pl50',
'pl60',
'pl70',
'pl80',
'pl90',
'pl110',
'pl120',
'il50',
'il60',
'il70',
'il80',
'il90',
'il100',
'il110',
'ps',
'pne']




'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    new_image = np.ones((640,640,3), np.uint8) * (2 ** 6)
    new_image = image
    image_data = np.array(new_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, print(s)  # assert min requirements met
    if verbose and not result:
        print(s)
    return result

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def main(args=None):
    # 加载dpu.bit
    overlay = DpuOverlay("dpu.bit")
    print("Finish read dpu.bit .")

    # 加载模型文件
    overlay.load_model("/home/ubuntu/yolov5s.xmodel")
    print("Finish load model .")


    dpu = overlay.runner
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    shapeOut0 = (tuple(outputTensors[0].dims))
    shapeOut1 = (tuple(outputTensors[1].dims))
    shapeOut2 = (tuple(outputTensors[2].dims))
    output_data = [np.empty(shapeOut0, dtype=np.int8, order="C"), 
                np.empty(shapeOut1, dtype=np.int8, order="C"),
                np.empty(shapeOut2, dtype=np.int8, order="C")]

    print("Finish generate input and output list .")
    
    path = '/home/ubuntu/test/images/*'
    
    time_list = [0, 0, 0]
    idx = 0
    for img_path in glob.glob(path):
        print(img_path)
        img = cv2.imread(img_path)
        
        t1 = time.time()
        
        # image_size = img.shape[:2]
        # image_data = np.array(pre_process(img, (640, 640)), dtype=np.float32)
        # image = input_data[0]
        # # Fetch data to DPU and trigger it
        # image[0,...] = image_data.reshape(shapeIn[1:])
        img = img[...,::-1]
        img = (torch.tensor(np.ascontiguousarray(img))/255).numpy()
        img = np.expand_dims(img, 0).astype(np.float32)
        input_data[0] = img
        t2 = time.time()
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
    
        t3 = time.time()

        # detect

        output = [
            torch.tensor(output_data[0]).float() / 4.,
            torch.tensor(output_data[1]).float() / 8.,
            torch.tensor(output_data[2]).float() / 4.,
        ]
        for i in range(3):
            output[i] = output[i].permute(0,3,1,2)
            
        # res = raw_post_process(output)
        # t4 = time.time()
        # time_list[0] += t2 - t1
        # time_list[1] += t3 - t2
        # time_list[2] += t4 - t3
        # idx = idx + 1
        # print(idx)
        # if(idx == 500):
        #     break
        
        
        #---------------------
        t1 = time.time()
        res = post_process(output)[0]
        t2 = time.time()
        print('post_process: ', (t2 - t1) * 1000)
        res[:, :4] = res[:, :4].round()
        
        
        im = cv2.imread(img_path)
        class_str=""
        for *xyxy, conf, cls in reversed(res):
            cv2.rectangle(im, (int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0),2)
            class_str += (class_name[int(cls.item())] + " ")
        print(class_str)
        cv2.putText(im, class_str,(10,630),cv2.FONT_HERSHEY_SIMPLEX,2,lineType=cv2.LINE_AA,color = (255,0,0),thickness = 3)
        outdir ="detect/"+ img_path.split('/')[-1]
        print(outdir)
        cv2.imwrite(outdir,im)
        
    # print(time_list[0]/idx*1000)
    # print(time_list[1]/idx*1000)
    # print(time_list[2]/idx*1000)
            
main()