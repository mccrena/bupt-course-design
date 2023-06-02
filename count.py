from subprocess import list2cmdline
import sys
from PIL import Image, ImageTk
from numpy import ndarray
from tkinter import *
from moviepy.editor import VideoFileClip
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from tkinter import filedialog
import tkinter as tk
########################################

from PIL import ImageTk, Image
import threading
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


output_dir = 'inference/output'  # 要保存到的文件夹
show_video = True  # 运行时是否显示
save_video = True  # 是否保存运行结果视频
save_text = True  # 是否保存结果数据到txt文件中，result.txt的格式是(帧序号,框序号,框到左边距离,框到顶上距离,框横长,框竖高,-1,-1,-1,-1)，number.txt的格式是(帧序号，直至当前帧跨过线的框数)
class_list = [0]  # 类别序号，在coco_classes.txt中查看（注意是序号不是行号），可以有一个或多个类别
big_to_small = 0  # 0表示从比线小的一侧往大的一侧，1反之
point_idx = 0  # 要检测的方框顶点号(0, 1, 2, 3)，看下边的图，当方框的顶点顺着big_to_small指定的方向跨过检测线时，计数器会+1
line = [0, 0, 1280, 0]  # 检测线的两个段点的xy坐标，总共4个数

iden = []
########################################
# 一些参数的定义
# x是点到左边的距离，y是点到顶上的距离
# 小于则说明点落在直线与x轴所夹的锐角区域

# 方框顶点的序号
#    0              1
#    |--------------|
#    |              |
#    |              |
#    |--------------|
#    3              2


#    |-------> x轴
#    |
#    |
#    V
#    y轴

########################################
# 一些数据处理

# x_i、y_i表示x、y在points数组中的下标
if point_idx == 0:
    x_i = 0
    y_i = 1
elif point_idx == 1:
    x_i = 2
    y_i = 1
elif point_idx == 2:
    x_i = 2
    y_i = 3
elif point_idx == 3:
    x_i = 0
    y_i = 3


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def point_bigger(line, x, y) -> bool:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    if y1 == y2:
        if y > y1:
            return True
        elif y <= y1:
            return False

    if x1 == x2:
        if x > x1:
            return True
        elif x <= x1:
            return False

    if (x - x1) / (x2 - x1) > (y - y1) / (y2 - y1):
        return True
    else:
        return False


def point_smaller(line, x, y) -> bool:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    if y1 == y2:
        if y < y1:
            return True
        elif y >= y1:
            return False

    if x1 == x2:
        if x < x1:
            return True
        elif x >= x1:
            return False

    if (x - x1) / (x2 - x1) < (y - y1) / (y2 - y1):
        return True
    else:
        return False


def judge_size(direction, line, x, y):
    if direction == 0:  # 从小到大
        return point_smaller(line, x, y)
    elif direction == 1:
        return point_bigger(line, x, y)
    else:
        print('方向错误，只能为0或1！')


########################################


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        


        blob = cv2.dnn.blobFromImage(im0, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  # 把裁剪后的人脸转换成神经网络可以处理的格式
        genderNet.setInput(blob)  # 把人脸输入性别检测神经网络
        genderPreds=genderNet.forward()  # 获取性别检测结果
        gender=genderList[genderPreds[0].argmax()]  # 根据检测结果确定性别
        ageNet.setInput(blob)  # 把人脸输入年龄检测神经网络
        agePreds=ageNet.forward()  # 获取年龄检测结果
        age=ageList[agePreds[0].argmax()]  # 根据检测结果确定年龄
        #new_boxes = [{'x1': fb[0], 'y1': fb[1], 'x2': fb[2], 'y2': fb[3]} for fb in faceBoxes]  # 将当前帧中所有人脸框的坐标信息转换为字典形式
        cv2.putText(im0, f'{gender}, {age}',  (x1, y1 + t_size[1] - 20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)  # 在视频帧上标注性别和年龄信息
    



        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
      #  cv2.putText(img, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    return img


# 在调用detect()函数进行检测时，记得加上
# with torch.no_grad():
#     detect(args)
def detect(opt):
    global im0
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    #####################################################
    # 参数设置
    show_vid = show_video
    save_vid = save_video
    save_txt = save_text

    #####################################################
    # 获取视频的信息
    a = cv2.VideoCapture(source)
    frame_num = int(a.get(7))  # 总帧数
    frame_rate = a.get(5)  # 帧速率
    frame_w = a.get(3)  # 帧宽
    frame_h = a.get(4)  # 帧高
    print(frame_num, frame_rate, frame_w, frame_h)
    a.release()

    #####################################################

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    ##################################
    print(device)
    ##################################
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    root.title("Video Stream")
    video_label = tk.Label(root)
    video_label.pack(padx=10, pady=10)
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()



        



        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)




                




                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                    iden.append(max(identities))
                    cv2.putText(im0, f'num = {max(iden)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)


            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))


            # Stream results
            if show_vid:
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                im0 = Image.fromarray(im0)
                im0 = im0.resize((1280, 720), Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=im0)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
                root.update()

def total():
   global vv
   vv = tk.filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

   if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=vv, help='source')
    parser.add_argument('--output', type=str, default=output_dir, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=class_list, type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    with torch.no_grad():
         detect(args)
    B.detect_video(vv,im0)
    
         
def stop_detection():
    img = Image.open("background.png")
    img = img.resize((800, 600), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(middle_frame, image=img)
    panel.pack(fill=tk.BOTH, expand=1)
        # 创建主窗口



root = tk.Tk()


root.title("人脸检测器")
root.geometry("1280x720")
#Top Frame
top_frame = tk.Frame(root, bg="#e6f0ff")
top_frame.pack(fill=tk.BOTH, expand=1)
# 设置 top_frame 大小
top_frame.config(height=10, width=20)


btn_import_video = tk.Button(top_frame, text="导入视频", font=("Arial", 16), command=total ) #
btn_import_video.pack(side=tk.LEFT, padx=5, pady=10)
btn_go_video = tk.Button(top_frame, text="停止检测", font=("Arial", 16), command=stop_detection)
btn_go_video.pack(side=tk.LEFT, padx=10, pady=10)



#Middle Frame

'''
middle_frame = tk.Frame(root, bg="#fff")
middle_frame.pack(fill=tk.BOTH, expand=1)
'''

#Bottom Frame
bottom_frame = tk.Frame(root, bg="#e6f0ff")
bottom_frame.pack(fill=tk.BOTH, expand=1)


root.mainloop()