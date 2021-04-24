import numpy as np
import cv2
import cv2.aruco as aruco
import math
# import matplotlib.pyplot as plt
import os
import time
from sys import argv


'''
REVISION PENDING
Use this script to receive live feed and find aruco patterns on it. Requires
a callibrated camera matrix and distance vector, which are for now hardcoded.
'''

# initialize the video recorder from IP camera
cap = cv2.VideoCapture('https://192.168.43.1:8085/video')
# cap = cv2.VideoCapture(0)

# Name of dump file to drop the stuff
filename =  str(argv[1])

def get_camera_pose(markerID=0):
    ''' Gets the camera pose with respect to a specific marker ID
    '''
    if [markerID] in ids:
        # find the marker id in the id list and get its tvec and rvec
        marker = np.where(ids == [markerID])
        tvec = tvecs[marker][0]
        rvec = rvecs[marker][0]

        # Get rotation matrix from object coordinates to camera coordinates
        rot = cv2.Rodrigues(rvec)[0]


        # Assemble translation matrix and use it to compute camera|markerID
        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3]   = tvec
        tvec_camera_from_camera = np.array([0,0,0,1])
        vector = np.matmul(np.linalg.inv(T), tvec_camera_from_camera)

        eul   = -cv2.decomposeProjectionMatrix(T[:3,:])[6]
        yaw   =  eul[1,0]
        pitch = (eul[0,0]+(90))*math.cos(eul[1,0])
        roll  = (-(90)-eul[0,0])*math.sin(eul[1,0]) +eul[2,0]

        return vector, rot, yaw, roll, pitch



# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]

'''# Substitute with manual calib (BEST UNTIL NOW, DO NOT OVERWRITE HARDCODED VALUES)
mtx = np.array([[1.84021751e+03, 0.00000000e+00, 5.42779058e+02],
               [0.00000000e+00, 2.39630743e+03, 9.54928911e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.04838364,  0.25314362,  0.00389218, -0.04914811, -0.14082424]])
'''


# Record image until stopped with Q
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # reset measurements
    x = float("nan")
    z = float("nan")
    yaw = float("nan")


    # Work out the aruco markers from the picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If some markers are found in the picture
    if corners != []:
        # define size of marker for scale [currently in mm]
        size_of_marker = 150

        # get the rotation and traslation vectors CAMERA -> ARUCO
        rvecs,tvecs,trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)

        ## Getting the pose of 1 w.r to 0
        # get_position_ID1_ID0()
        try:
            vector, rot, yaw, roll, pitch = get_camera_pose()
        except:
            pass

        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n Corners \n', corners)
        print('\n ids \n', ids)
        print('\n Rvecs \n', rvecs)
        print('\n Tvecs \n', tvecs)
        print('\n Mtx \n', mtx)
        print('\n Dist \n', dist)
        print('\n Camera coordinates: \n', vector)
        print('\n Camera rotation matrix: \n', rot)
        print('\nYaw, roll, pitch:\n ', yaw, roll, pitch)

        # draw aruco markers on pic
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # draw aruco 3D axes on picture
        for i, j in zip(rvecs, tvecs):
            frame = aruco.drawAxis(frame, mtx, dist, i, j, 50)


    # Display the resulting frame, pyrDown so it fits the screen
    cv2.imshow('frame',cv2.pyrDown(frame))
    # cv2.imshow('frame', frame)

    # stop looping on Q press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    x = vector[0]
    z = vector[2]
    with open(filename, 'a') as file:
        file.write(f'{time.time()} {x} {z} {yaw}\n')

# Hold the capture until a second keypress is made, then close
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
