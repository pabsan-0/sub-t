import cv2
import numpy as np
import ctypes
import time
import os
import sys

'''
This python script will initialize darknet from python command bindings and
retrieve feed from a IP camera source, which is to be inferred in real time.

In this version, the target item [bottle] is detected and a discrete map with
origin on the camera is plotted, the regions in white denoting the position
of the bottle, which is updated in real-time.
'''



# instance the video capture
cap = cv2.VideoCapture('https://192.168.0.100:8085/video')

# Items to detect + expected width [mm]
item_width = {'bottle': 100}

# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]




# VERY NASTY way of importing the darknet bindings without installing extra packages
os.chdir('/home/pablo/YOLOv4/darknet')
sys.path.insert(1, '/home/pablo/YOLOv4/darknet')
try:
    import darknet
    import darknet_images
    from darknet import *
except OSError:
    # deals with the LD library path for CUDNN, run once per terminal execution
    print('''>>> Catched CUDNN error! Run this bash command and try again:\n
    \rexport LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib
    ''')
    quit()




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


def locate_item():
    '''
    This function allows for the computation of the position of an item with
    respect to the camera. Adapted from the demo function in this repository.
    '''
    # Dirty ad-hoc way to get positional information about example item
    try:
        # checks if the item has been detected (returns boolean pointer [T,F,T...])
        ii = ['bottle' in i[0] for i in detections]

        # retrieve the position of the first True in a generator fashion
        y = (i for i,v in enumerate(ii) if v==True)
        ii = next(y)

    except StopIteration:
        return None, None, None

    if ii == []:
        return None, None, None

    # Adjust the scaling factor, input_width=800 but darknet_width=416
    # print(image_out.shape[1])
    # print(frame.shape[1])
    rescaling_factor = frame.shape[1] / image_out.shape[1]
    u1 = (detections[ii][2][0] - detections[ii][2][2] /2) * rescaling_factor
    u2 = (detections[ii][2][0] + detections[ii][2][2] /2) * rescaling_factor
    dx = item_width['bottle']

    # Perform math to get results (see README)
    fx = mtx[0,0]
    cx = mtx[0,2]
    s = fx * (dx) / (u2 - u1)
    A = (u1 - cx)/(u2 - cx)
    x1 = (A * dx) / (1 - A)
    x2 = x1 + dx
    z = (fx * x1 - s * u1) / cx

    return z, x1, x2


class worldMap(object):

    def __init__(self):
        # matrix holding the discrete map to be drawn
        self.map = np.zeros([32, 32])

        # bins used for the conversion of continuous points to discrete map
        self.bins = np.array([100 * i -1500 for i in range(31)])

        # matrix holding the mask used for false positive removal
        self.mask_removal = np.ones(self.map.shape) * 0.05


    def update(self, z, x):
        # increase probability for true positives
        z_i, x_i = np.digitize([z,x], self.bins)
        self.map[z_i][x_i] += 0.5

        # decrease probability for false negatives that are recognized now
        self.map = self.map - self.mask_removal

        # bound values
        self.map = np.clip(self.map, 0, 2)




if __name__ == '__main__':
    # load a nnet params
    config_file = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.cfg'
    data_file = '/home/pablo/YOLOv4/darknet/cfg/coco.data'
    weights = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.weights'

    # initialize network
    network, class_names, colors = load_network(config_file, data_file, weights, batch_size=1)

    # Instance the world map
    map = worldMap()

    # main loop
    decimator = 0
    while 1:
        # load an image
        ret, frame = cap.read()
        frame = cv2.undistort(frame, mtx, dist, None)

        # dont use all frames so that the video runs more smoothly
        if decimator % 50:
            decimator = 0

            # infer frame
            image_out, detections = image_detection(frame, network, class_names, colors, .5)

            # print the dictionary with current detections
            if detections != []:
                # print('Removing all items not in: ',  item_width.keys())
                detections = [i for i in detections if 'bottle' in i[0]]

            z, x1, x2 = locate_item()
            if [z, x1, x2] == [None, None, None]:
                continue

            # Printout real world coordinates
            print('Computed distance to item [mm] ', z)
            print('Computed center ', (x1 + x2)/2)

            map.update(z, (x1 + x2)/2)

            # display image. waitkey(1) is required to give imshow time to render
            cv2.imshow('live', image_out)
            cv2.imshow('map', cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(map.map))))
            cv2.waitKey(1)

        decimator += 1
