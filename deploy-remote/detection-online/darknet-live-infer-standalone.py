import cv2
import numpy as np
import ctypes
import time
import os
import sys

'''
This python script will initialize darknet from python command bindings and
retrieve feed from a IP camera source, which is to be inferred in real time. For
this, darknet must be installed and the cwd moved to its install folder (edit
hardcoded paths).
'''

# VERY NASTY way of importing the darknet bindings without installing extra packages
os.chdir('/home/pablo/YOLOv4/darknet')
sys.path.insert(1, '/home/pablo/YOLOv4/darknet')
try:
    import darknet
    import darknet_images
    from darknet import *
except OSError:
    # deals with the LD library path for CUDNN, run once per terminal execution
    print(''' Catched CUDNN error! Run this bash command and try again:
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib
    ''')
    quit()


# instance the video capture
cap = cv2.VideoCapture('https://192.168.0.101:8085/video')




def image_detection(image, network, class_names, class_colors, thresh):
    '''
    Modified version of the function from darknet_images.py that accepts
    a live frame instead of a file path for loading an image, conveting it to
    C format and performing Inference.
    '''
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



# load a nnet params
config_file = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.cfg'
data_file = '/home/pablo/YOLOv4/darknet/cfg/coco.data'
weights = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.weights'

# initialize network
network, class_names, colors = load_network(config_file, data_file, weights, batch_size=1)

# main loop
decimator = 0
while 1:
    # load an image
    ret, frame = cap.read()

    # dont use all frames so that the video runs more smoothly
    if decimator % 5:
        # infer frame
        image_out, detections = image_detection(frame, network, class_names, colors, .5)

        # print the dictionary with current detections
        print(detections)

        # display image. waitkey(20) is required to give imshow time to render
        # IF EXPERIENCING SEGMENTATION FAULTS COMMENT THESE TWO LINES
        cv2.imshow('live', image_out)
        cv2.waitKey(20)

    decimator += 1
