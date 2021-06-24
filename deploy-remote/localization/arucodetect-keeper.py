import numpy as np
import cv2
import cv2.aruco as aruco
import math
# import matplotlib.pyplot as plt
import os
import time
import pprint

'''
Use this script to receive live feed and find aruco patterns on it.
Requires a callibrated camera matrix and distortion vector.

DEPS: RedmiNote9Pro.nz: a file from which to load camera distortion matrix
'''

# initialize the video recorder from IP camera
cap = cv2.VideoCapture('https://192.168.43.1:8085/video')
# cap = cv2.VideoCapture(0)



def get_position_ID1_ID0():
    ''' Gets the pose of the markerID1 with respect of the markerID2
    '''
    if [0] in ids and [1] in ids:
        # check where 0 and 1 are in the 'ids' list
        zero = np.where(ids == [0])
        one = np.where(ids == [1])

        # Get traslation and rotation vectors, plus rotation matrix for ID0
        tvec_0 = tvecs[zero][0]
        rvec_0 = rvecs[zero][0]
        rot_0 = cv2.Rodrigues(rvec_0)[0]

        # get traslation and rotation vectors for ID1
        tvec_1 = tvecs[one][0]
        rvec_1 = rvecs[one][0]

        # Assemble translation matrix and use it to compute R1|R0
        T = np.eye(4)
        T[0:3, 0:3] = rot_0
        T[0:3, 3]   = tvec_0
        tvec_1 = np.append(tvec_1, [1])
        vector = np.matmul(np.linalg.inv(T), tvec_1)

        # print and return answer, croppting the bureaucratic '1' from the vector
        print('\n Coordinates of markerID1 | markerID0: \n', vector[:-1])
        return vector[:-1]
    else:
        return None



def get_camera_pose_old(markerID=0):
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

        return vector, yaw, roll, pitch


def get_camera_pose(markerID=0):
    ''' Gets the camera pose with respect to any set of detected IDs
    '''
    # Define placeholder for detection data
    detections = {}

    for idx, markerID in enumerate(ids):
        # get the tvexs and rvecs of this particular marker
        tvec = tvecs[idx][0]
        rvec = rvecs[idx][0]

        # Get rotation matrix from object coordinates to camera coordinates
        rot = cv2.Rodrigues(rvec)[0]

        # Assemble translation matrix and use it to compute camera|markerID
        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3]   = tvec
        tvec_camera_from_camera = np.array([0,0,0,1])
        x, y, z, _ = np.matmul(np.linalg.inv(T), tvec_camera_from_camera)

        # the euler angle is the Projection matrix Aruco/camera, not camera/aruco !!!!!!!
        eul   = -cv2.decomposeProjectionMatrix(T[:3,:])[6]
        yaw   =  eul[1,0]
        pitch = (eul[0,0]+(90))*math.cos(eul[1,0])
        roll  = (-(90)-eul[0,0])*math.sin(eul[1,0]) + eul[2,0]

        # Add each marker's pose to the detection dictionary
        detections[idx] =  {#'raw': [int(markerID), x, y, z, yaw, roll, pitch],
                            'id': int(markerID),
                            'pose': [x, z, yaw]}

    return detections




# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]

# Record image until stopped with Q
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # reset measurements to avoid issues (empyrical, no formal justification)
    x = float("nan")
    z = float("nan")
    vector = [float("nan"), float("nan"), float("nan") ]
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

        # Get camera pose with respect to all markers
        detections = get_camera_pose()

        # Terminal display
        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n Poses (x, z, yaw)')
        pprint.pprint(detections, width=1)

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


# Hold the capture until a second keypress is made, then close
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
