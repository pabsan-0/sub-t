# move to darknet and call python interpreter by specifying the CUDNN shortcut
cd ~/YOLOv4/darknet

cat << EOF > demo_pablo_pythonscript.py
import cv2
import numpy as np
import ctypes

cap = cv2.VideoCapture('https://192.168.0.101:8085/video')

def numpy2darknet_img(image): # v1
    ''' STOLEN FROM https://github.com/pjreddie/darknet/issues/423 '''
    image = (image / 255.).astype(np.float32)
    assert (image.shape[2] == 3)
    data = np.swapaxes(image, 2, 1)
    data = np.swapaxes(data, 1, 0)
    print("type: ", data.shape, data.dtype)
    data = np.ascontiguousarray(data)
    im = IMAGE(w=image.shape[1], h=image.shape[0], c=3, data=data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return im

# import the python wrapper commands
from darknet import *

# load a nnet
config_file = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.cfg'
data_file = '/home/pablo/YOLOv4/darknet/cfg/coco.data'
weights = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.weights'

network, class_names, colors = load_network(config_file, data_file, weights, batch_size=1)

while 1:
    # load an image
    ret, frame = cap.read()
    image = numpy2darknet_img(frame)

    # infer
    results = detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45)
    print(results)

EOF

LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib python3 demo_pablo_pythonscript.py

cd -
