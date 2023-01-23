from django.shortcuts import render
from django.http import StreamingHttpResponse
import yolov5, torch
from yolov5.utils.general import (check_img_size, non_max_suppression, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import cv2
from PIL import Image as im
from django.contrib import messages

# Create your views here.
def login(request):
    return render(request, 'login.html')



def index(request):
    return render(request, 'index.html')



#load model
model = yolov5.load('yolov5s.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('0') # 0 for gpu, '' for cpu
# initialize deepsort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25',
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
 
def stream(request):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break
        
        results = model(frame, augment=True)
        # process
        for i in results.render():
            data = im.fromarray(i)
            data.save('demo.jpg')
            cv2.imwrite('demo.jpg', frame)

        if len(results.xyxy[0]) > 0:
            
            print(' detected')
        #    return {'mi_variable': 'Hola mundo'}
        else:
            print('non detected')
       #     return {'mi_variable': 'Hola mundo'}
       
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +  open('demo.jpg', 'rb').read() + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(stream(request), content_type='multipart/x-mixed-replace; boundary=frame')

def hola():
    return {'mi_variable': 'Hola mundo'}