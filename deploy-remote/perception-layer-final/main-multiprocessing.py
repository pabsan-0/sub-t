################################################################################
#### main-multiprocessing.py                                                ####
#### Pablo Santana - CNNs for object detection in subterranean environments ####
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │                 │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#  └─YOU'RE HERE│   p3                   p4                   │
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#
# Run with ``$ python3 main-multiprocessing.py`
# Make sure all submodules and assets are available to the main process.

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
import pabloKalman_yawEuclidean as pabloKalman
# import pabloKalman as pabloKalman

import pabloDarknetInference
import pabloItem2CameraPosition
import pabloWorldMap

from pabloCameraPoseAruco import ArucoMarker
from pabloKalman_yawEuclidean import KalmanFilter
# from pabloKalman import KalmanFilter
from pabloWorldMap import worldMap

from multiprocessing import Process, Queue


def trackingMessage(id, parentName, outputMetricName, outputMetric, now):
    """
    Used for displaying compact messages of time and output to track the
    performance of each parallel process. Mainly a debugging tool.
    """
    if timeAllModules and not showAllModulesOutput:
        # Display elapsed time per module processing
        str_out = f'{parentName} took {time.time()-now} seconds'

    elif showAllModulesOutput:
        # Format result for pretty printing
        pretty_result_string = pprint.pformat(outputMetric)

        # Print all in a single message to avoid process print overlapping
        str_out = f'''\n\n{id * 100}
        \r\r\r>>> Module {parentName} says: ({time.time()})')
        \r\r\r>>  Elapsed time: {time.time()-now} seconds.
        \r\r\r>>  {outputMetricName} value: \n{pretty_result_string}
        '''
    else:
        str_out = ''
    return str_out



def parallelCameraPoseAruco(markerDict, cameraMatrix, qframeUndist_2aruco, qCamera2ArUcoPose, qPlotland_1, qLocalizationSuccess):
    """
    Get camera pose from ArUco markers. Returns dict with each marker data.
    """
    # Aruco dict. Selecting subset avoids running into unexpected IDs!
    # This should be working but isnt!! cant generate small size dict from 250x250
    num_markers = len(markerDict)                       # Bold hypotheses
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    aruco_dict = cv2.aruco_Dictionary.create_from(num_markers, 4, aruco_dict)
    print('LOOK HERE, we are using this # markers ',num_markers)

    # Required aruco params (default, braindead copypaste)
    parameters =  aruco.DetectorParameters_create()

    # Infer which markers are susceptible to be found in the world
    env_markers = [markerDict[i].id for i in markerDict.keys()]


    # Hardcoded marker size and fake distortion coefs (for usage on undist image)
    fakeDist = np.array([[1e-5,1e-5,1e-5,1e-5,1e-5]])
    size_of_marker = 150

    # Dummy fake detection to cover if the first iteration fails to find an ArUco
    camera2aruco_pose = {}
    camera2aruco_pose[0] = {
                        'id':0,
                        'pose2aruco': np.array([0,0,0]),
                        'pose2world': np.array([0,0,0]),
                        }

    while 1:
        # time tracker
        now = time.time()

        # Get an undistorted frame from the input queue
        frameUndist = qframeUndist_2aruco.get()

        # Try to estimate camera pose
        try:
            camera2aruco_pose, frameOutArUco = pabloCameraPoseAruco.main(
                frameUndist,
                aruco_dict, parameters,
                env_markers, size_of_marker,
                cameraMatrix, fakeDist,
                markerDict,
                debug=False
                )
            LocalizationSuccess = 1
        except pabloCameraPoseAruco.CorruptedArUcoDetections:
            # Either no arucos in sight or an unexpected one: return previous pose
            frameOutArUco = frameUndist
            LocalizationSuccess = 0

        # Send estimation through output queue
        qCamera2ArUcoPose.put(camera2aruco_pose)

        # Send image and success flag through output aux queue
        qPlotland_1.put(frameOutArUco)
        qLocalizationSuccess.put(LocalizationSuccess)

        # Time and output tracking (enable/disable) at MAIN
        print(trackingMessage('1', 'CameraPoseAruco', 'camera2aruco_pose', camera2aruco_pose, now), end='')



def parallelKalman(markerDict, qCamera2ArUcoPose, qCameraPoseFiltered):
    """
    KalmanFilters a live Pose input from multiple ArUcos at once
    """
    # Import model matrices (hardcoded)
    A, B, C, _, Rw, Rv = pabloKalman.modelImport(dataSample='data1.txt')

    # Instance the handmade Kalman filter
    KF = KalmanFilter(A, B, C, [], Rw, Rv)

    while 1:
        # time tracker
        now = time.time()

        # Get unfiltered pose dict from input queue
        camera2aruco_pose = qCamera2ArUcoPose.get()

        # Kalman-filter the current measurement
        cameraPoseFiltered = pabloKalman.main(
            KF,
            camera2aruco_pose,
            markerDict
            )

        # Drop filtered pose dict to output queue
        qCameraPoseFiltered.put(cameraPoseFiltered)

        # Time and output tracking (enable/disable) at MAIN
        print(trackingMessage('2', 'KalmanFilter', 'cameraPoseFiltered', cameraPoseFiltered, now), end='')



def parallelDarknetInference(qframeUndist_2darknet, qItemDetections, qPlotland_2):
    """
    Performs darknet inference on image via python bindings
    """
    # Change dirs for darknet importing
    sys.path.append(os.getcwd())
    os.chdir('/home/pablo/YOLOv4/darknet')
    sys.path.insert(1, '/home/pablo/YOLOv4/darknet')

    # Initialize the network
    network, class_names, class_colors = pabloDarknetInference.networkBoot()

    while 1:
        # time tracker
        now = time.time()

        # Get undistorted frame from input queue
        frameUndist = qframeUndist_2darknet.get()

        # Image detection with darknet!
        frameDetected, itemDetections = pabloDarknetInference.imageDetection(
            frameUndist,
            network,
            class_names,
            class_colors,
            thresh=0.3
            )

        # Drop detected + input frames plus detection output to output queue
        qItemDetections.put([frameDetected, frameUndist, itemDetections])

        # Send image through output aux queue
        qPlotland_2.put(frameDetected)

        # Time and output tracking (enable/disable) at MAIN
        print(trackingMessage('3','DarknetInference', 'itemDetections', itemDetections, now), end='')



def parallelItem2CameraPosition(itemListAndSize, cameraMatrix, qItemDetections, qItem2CameraPosition):
    """
    Using an item's width and given bounding boxes, infer its position wrt camera
    """
    while 1:
        # time tracker
        now = time.time()

        # Get detected + input frames plus detection output input queue
        frameDetected, frameUndist, itemDetections = qItemDetections.get()

        # Get the horizontal position of the item with respect to the camera
        item2CameraPosition = pabloItem2CameraPosition.getItemPosition(
            frameDetected,
            frameUndist,
            itemDetections,
            itemListAndSize,
            cameraMatrix
            )

        # Drop item position to output queue
        qItem2CameraPosition.put(item2CameraPosition)

        # Time and output tracking (enable/disable) at MAIN
        print(trackingMessage('4','item2CameraPosition', 'item2CameraPosition', item2CameraPosition, now), end='')



def parallelWorldMap(itemListAndSize, qCameraPoseFiltered, qItem2CameraPosition, qPlotland_1, qPlotland_2, qLocalizationSuccess):
    """
    Given camera Pose and object positions with respect to it, draw a world Map
    in which a likelihood map for the presence of an item at a given location is
    shown.
    """

    # Instance the worldMap class
    map = worldMap(
        numItems=len(itemListAndSize),
        mapsize_XZ_cm=[84+277,676]
        )

    # Small-time configuration of dissapearance and appearance rates
    map.disappearanceRate = 0.0002
    map.appearanceRate    = 0.0005

    # adjust worlFrame Coordinates wr to map origin
    map.world2map_XZ_cm = [84, 0]

    while 1:
        # time tracker
        now = time.time()

        # Get item and camera poses from input queues
        cameraPoseFiltered = qCameraPoseFiltered.get()
        item2CameraPosition = qItem2CameraPosition.get()

        # Get data from auxiliary queues
        frameArUco = qPlotland_1.get()
        frameDetected = qPlotland_2.get()
        LocalizationSuccess = qLocalizationSuccess.get()

        # if localization has been successful do map detections else dont
        if LocalizationSuccess == 1:
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



            # Show if localization is okay else freeze
            cv2.imshow('Field of view', map.fovMask)
            cv2.imshow('Instantaneous discovery', np.array(map.discoveryMask, np.uint8))
            #cv2.imshow('Likelihood map', map.map)
            cv2.imshow('Likelihood map', map.map_fov)

        # Always show
        cv2.imshow('ArUco pose estimator', frameArUco)
        cv2.imshow('Detection', frameDetected)
        cv2.waitKey(1)

        if timeAllModules:
            print(f'\tMapping took {time.time()-now} seconds')



if __name__ == '__main__':

    # Do you want process time's being shown? ATT. LOTS OF PRINTOUTS
    timeAllModules, showAllModulesOutput = False, True

    # Video feed source
    cap = cv2.VideoCapture('https://192.168.0.102:8085/video')

    # Load camera calibration matrices
    with np.load('RedmiNote9Pro.npz') as X:
        cameraMatrix, distCoeffs = [X[i] for i in('cameraMatrix', 'distCoeffs')]
    fakeDist = np.array([[1e-5,1e-5,1e-5,1e-5,1e-5]])

    # Kalman Rv, preimported here to assign Rv to markerDict, again on Kalman module
    #_,_,_,_,_, Rv = pabloKalman.modelImport(dataSample='data1.txt')
    # Use this to solve non-euclidean angle issue
    Rv = np.eye(4)

    # Marker coordinates in the world, inc Kalman Noise matrix
    # angle measured in negative y!
    markerDict_BedroomCorner = {
        0: ArucoMarker(0, [    0,    0,   0], Rv),
        1: ArucoMarker(1, [ -640,    0,   0], Rv),
        2: ArucoMarker(2, [-1100,  370,  -90], Rv),
        }

    markerDict_LivingRoom = {
        0: ArucoMarker(0, [    0,    0,   0], Rv),
        1: ArucoMarker(1, [ 1500,    0,   0], Rv),
        2: ArucoMarker(2, [ 2770, 1260,  90], Rv),
        3: ArucoMarker(3, [ 2770, 3070,  90], Rv),
        4: ArucoMarker(4, [ 2770, 4980,  90], Rv),
        5: ArucoMarker(5, [ 2770, 6200,  90], Rv),
        6: ArucoMarker(6, [ 1400, 6760, 180], Rv),
        7: ArucoMarker(7, [ -240, 4890, -90], Rv),
        8: ArucoMarker(8, [ -440, 4570, 180], Rv),
        9: ArucoMarker(9, [ -840, 2740, -90], Rv),
        }

    # Select world. REMEMBER TO CHANGE MAP SIZE
    markerDict = markerDict_LivingRoom


    # list of items of interest and their estimate width
    itemListAndSize = {'bottle': 100
            }

    # Main data queues
    q1 = qframeUndist_2aruco     = Queue(maxsize=1)
    q2 = qframeUndist_2darknet   = Queue(maxsize=1)
    q3 = qCamera2ArUcoPose       = Queue(maxsize=1)
    q4 = qCameraPoseFiltered     = Queue(maxsize=1)
    q5 = qItemDetections         = Queue(maxsize=1)
    q6 = qItem2CameraPosition    = Queue(maxsize=1)

    # Auxiliar data bypassing
    # Move detected frame to be plotted (cv2 cant handle multiprocess imshow)
    q7 = qPlotland_1             = Queue(maxsize=1)
    q8 = qPlotland_2             = Queue(maxsize=1)
    #
    # Successful location frame to not print detections if cameraPose ambiguous
    q9 = qLocalizationSuccess    = Queue(maxsize=1)


    # Defining the processes. Placeholders to be able to check on them from mains
    p1 = Process(target=parallelCameraPoseAruco,
                 args=(markerDict, 
                       cameraMatrix, 
                       qframeUndist_2aruco, 
                       qCamera2ArUcoPose, 
                       qPlotland_1, 
                       qLocalizationSuccess,))
    
    p2 = Process(target=parallelKalman,
                 args=(markerDict, 
                       qCamera2ArUcoPose, 
                       qCameraPoseFiltered,))
    
    p3 = Process(target=parallelDarknetInference,
                 args=(qframeUndist_2darknet, 
                       qItemDetections, 
                       qPlotland_2,))
    
    p4 = Process(target=parallelItem2CameraPosition,
                 args=(itemListAndSize, 
                       cameraMatrix, 
                       qItemDetections, 
                       qItem2CameraPosition,))
    
    p5 = Process(target=parallelWorldMap,
                 args=(itemListAndSize, 
                       qCameraPoseFiltered, 
                       qItem2CameraPosition, 
                       qPlotland_1, 
                       qPlotland_2, 
                       qLocalizationSuccess,))

    # Start all the processes
    for p in [p1,p2,p3,p4,p5]:
        p.start()

    # For FPS tracking (if option selected under __main__)
    now = time.time()

    while 1:
        # Read a frame from the capture object correct image distortions
        ret, frameIn = cap.read()
        frameUndist = cv2.undistort(frameIn, cameraMatrix, distCoeffs, None)

        # This allows real-time frames without clogging the queues
        # Once these queues hold values the rest of the pipeline starts working
        if qframeUndist_2aruco.empty() & qframeUndist_2darknet.empty():
            qframeUndist_2aruco.put(frameUndist)
            qframeUndist_2darknet.put(frameUndist)

            # Just for time tracking (if option selected under __main__)
            if timeAllModules:
                print(f'Doing {1/(time.time() - now)} FPS!')
                now = time.time()

            # Make sure that the CUDNN issue solution is cleanly shown on exception
            # This will allow the error-solution message to be seen easily.
            if not p3.is_alive():
                print('Run the previous command on this terminal. \
                      Press ^C (ctrl+C) to exit and ignore the error message that will pop up.')
                quit()
