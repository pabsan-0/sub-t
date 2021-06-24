import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import cv2
import cv2.aruco as aruco
import math
import os
import time
import pprint
import psutil
import ctypes
import sys

import pabloCameraPoseAruco
import pabloKalman
import pabloDarknetInference
import pabloItem2CameraPosition
import pabloWorldMap

from pabloCameraPoseAruco import ArucoMarker
from pabloKalman import KalmanFilter
from pabloWorldMap import worldMap



# Video feed source
cap = cv2.VideoCapture('https://192.168.0.103:8085/video')

# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]
fakeDist = np.array([[1e-5,1e-5,1e-5,1e-5,1e-5]])

# Import model matrices (hardcoded) and instance kalman filter
A, B, C, _, Rw, Rv = pabloKalman.modelImport(dataSample='data1.txt')
KF = KalmanFilter(A, B, C, [], Rw, Rv)

# Defining all the aruco-related stuff
# marker map (implementation supports adding kalman Rv, but omitted here)
markers = {
    0: ArucoMarker(0, [    0,    0,   0], Rv),
    1: ArucoMarker(1, [ -640,    0,   0], Rv),
    2: ArucoMarker(2, [-1100,  370,  -90], Rv),
    }
env_markers = [markers[i].id for i in markers.keys()]
size_of_marker = 150
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters =  aruco.DetectorParameters_create()


sys.path.append(os.getcwd())
os.chdir('/home/pablo/YOLOv4/darknet')
sys.path.insert(1, '/home/pablo/YOLOv4/darknet')
network, class_names, class_colors = pabloDarknetInference.networkBoot()

itemListAndSize = {'bottle': 100
        }

map = worldMap(
    numItems=len(itemListAndSize)
    )


while 1:
    # load and undistort input frame
    ret, frameIn = cap.read()
    frameUndist = cv2.undistort(frameIn, mtx, dist, None)

    # Get camera pose from ArUco markers. Returns dict with each marker data.
    try:
        camera2aruco_pose, frameOutArUco = pabloCameraPoseAruco.main(
            frameUndist,
            aruco_dict, parameters,
            env_markers, size_of_marker,
            mtx, fakeDist,
            markers,
            debug=False
            )
    except pabloCameraPoseAruco.CorruptedArUcoDetections:
        frameOutArUco = frameIn
        pass


    # Multi-aruco discrete Kalman Filter for noisy poses.
    cameraPoseFiltered = pabloKalman.main(
        KF,
        camera2aruco_pose,
        markers
        )


    # Image detection with darknet!
    frameDetected, itemDetections = pabloDarknetInference.imageDetection(
        frameUndist,
        network,
        class_names,
        class_colors,
        thresh=0.3
        )


    # Get the horizontal position of the item with respect to the camera
    item2CameraPosition = pabloItem2CameraPosition.getItemPosition(
        frameDetected,
        frameUndist,
        itemDetections,
        itemListAndSize,
        mtx
        )

    # Fix coordinate frame consistency issues. Units to mm-deg
    item2CameraPositionCompliant, cameraPoseFilteredCompliant \
        = pabloWorldMap.cameraCoordinateComply(
            item2CameraPosition,
            cameraPoseFiltered
            )

    # Update the map with detections!
    map.update(
        item2CameraPositionCompliant,
        cameraPoseFilteredCompliant
        )

    # its the imshows that have a big repercussion on latency!
    cv2.imshow('1', cv2.pyrDown(frameOutArUco))
    cv2.imshow('frame detected', frameDetected)
    cv2.waitKey(5)
    os.system('clear')
    #
    print('\n\n#### MEASURED METRICS ###')
    print('\nCamera pose before kalman X-Z-YAW [mm-deg]'); pprint.pprint(camera2aruco_pose, width=1)
    print('\nCamera pose after kalman X-Z-YAW [mm-deg]'); pprint.pprint(cameraPoseFiltered)
    print('\nItem position wrt camera Z-X [mm]'); pprint.pprint(item2CameraPosition)
    print('\n\n#### COORDINATE COMPLIANT METRICS ###')
    print('\nCompliant camera2world pose X-Z-YAW [mm-deg]'); print(cameraPoseFilteredCompliant)
    print('\nCompliant Item position wrt camera Z-X [mm]'); print(item2CameraPositionCompliant)
    # print('\n\n#### WORLD MAP METRICS ###')
    # print('\nFoV triangle [ABC], (C==CAMERA)'); print(abc2map)
    # print('\nItem2map position'); print(p2w)

# thread 2
# camera infers items
# compute item2camera_pos

#join threads
# draw worldmap
