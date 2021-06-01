################################################################################
#    main-multiprocessing.py / pabloCameraPoseAruco                            #
#    Pablo Santana - CNNs for object detection in subterranean environments    #
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │                 │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#               │   p3   └─YOU'RE HERE   p4                   │
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#

import numpy as np
import cv2
import cv2.aruco as aruco
import math
import os
import pprint


class CorruptedArUcoDetections(Exception):
    def __init__(self, message):
        self.message = message


class ArucoMarker(object):
    '''
    Implements a marker object to store its pose and kalman gain.

    ARGS:
        id:   ID of this ARUCO marker.
        pose: Provide as list [x, z, yaw]. Pose of this ARUCO marker.
        Rv:   Measurement noise covariance matrix of the camera seeing this ID.
    '''

    def __init__(self, id, pose, Rv=[]):
        # Store marker ID, pose
        self.id = id
        self.pose = pose

        # Store Kalman gain and Measurement Noise Covariance
        #self.L = np.zeros([6, 3], dtype=np.float32)
        # IF USING COS-SIN KALMAN FILTER
        self.L = np.zeros([4, 4], dtype=np.float32)
        self.R = Rv

        # Compute and store the rotation matrix of this marker
        x = pose[0]
        z = pose[1]
        yaw = np.deg2rad(pose[2])
        self.rot = np.array([[np.cos(yaw), -np.sin(yaw),   x],\
                             [np.sin(yaw),  np.cos(yaw),   z],\
                             [          0,            0,   1]])

    def transform2D_toWorld(self, pose_camera2marker):
        '''
        Applies homogeneous 2D transform to project pose of camera with
        respect to a marker and obtain the pose of the camera with respect to
        the origin coordinate frame.
        '''

        # Convert input [x, z, yaw] to [x, z, 1] for 2D conversion
        B = pose_camera2marker * np.array([1,1,0]) + np.array([0,0,1])

        # Apply homogeneous transformation matrix to get [x, z] w.r to origin
        A = self.rot @ B

        # Add angle camera_/marker + marker_/origin
        yaw = pose_camera2marker[2] + self.pose[2]

        # Recover yaw angle and return array in form [x, z, yaw]
        # returning   X & Z Term     +       angle
        pose_camera2world = A * np.array([1,1,0]) + yaw * np.array([0,0,1])

        return pose_camera2world




def detectMarkers(frame, aruco_dict, parameters, env_markers, size_of_marker, mtx, dist):
    ''' Use to compute the translation and rotation vectors of ArUcos in an
    image, including their 2D projections (corners)
    '''
    # Detect ArUcos from grayscaled picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Filter out markers that are not desired to be detected.
    # this is a consequence of poor dictionary choice, but I already got...
    # ... printed arucos for dict 250x250 and i only wanna use the first 10
    try:
        corners = [cor for i,cor in enumerate(corners) if int(ids[i][0]) in env_markers]
        ids = np.array([i for i in ids if int(i[0]) in env_markers])
    except:
        pass

    # If some markers are found in the picture
    if corners != []:
        '''if not all([i in env_markers for i in ids]):
            # If a marker that is not expected appears (noise) raise exception
            # this situation should not happen if aruco_dict is properly defined
            raise CorruptedArUcoDetections('Unexpected ArUco marker spotted.')'''

        # get the rotation and traslation vectors CAMERA -> ARUCO
        rvecs,tvecs,trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)
        return corners, ids, rvecs, tvecs

    else:
        raise CorruptedArUcoDetections('No ArUcos being detected.')


def drawDetectedMarkers(frame, corners, ids, rvecs, tvecs, mtx, dist):
    ''' Draw aruco markers and 3D axes on picture
    '''
    frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    for i, j in zip(rvecs, tvecs):
        frame = aruco.drawAxis(frame, mtx, dist, i, j, 50)
    return frame


def getCamera2MarkerPose(ids, rvecs, tvecs, markers):
    ''' Gets the camera pose with respect each marker in any set of detected IDs.
    '''
    # Define placeholder for detection data
    pose_camera2marker = {}

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

        eul   = -cv2.decomposeProjectionMatrix(T[:3,:])[6]
        yaw   =  eul[1,0]
        # pitch = (eul[0,0]+(90))*math.cos(eul[1,0])
        # roll  = (-(90)-eul[0,0])*math.sin(eul[1,0]) + eul[2,0]

        # Add each marker's pose to the detection dictionary
        # transform2D to relate the position wr this aruco to the world frame
        pose_camera2marker[idx] =  {#'raw': [int(markerID), x, y, z, yaw, roll, pitch],
                            'id': int(markerID),
                            'pose2aruco': np.array([x, z, yaw]),
                            'pose2world': markers[int(markerID)].transform2D_toWorld(np.array([x, z, yaw]))
                            }

    return pose_camera2marker



# CALL FROM LOOP
def main(frame, aruco_dict, parameters, env_markers, size_of_marker, mtx, dist, markers, debug=False):
    # placeholder in case of corner break later
    camera2aruco_pose = None

    # get the arucos spatial information, if error-free then compute camera poses
    corners, ids, rvecs, tvecs = detectMarkers(frame, aruco_dict, parameters, env_markers, size_of_marker, mtx, dist)
    camera2aruco_pose = getCamera2MarkerPose(ids, rvecs, tvecs, markers)

    # draw and output a picture with the detected markers
    frameOut = drawDetectedMarkers(frame, corners, ids, rvecs, tvecs, mtx, dist)

    # terminal + video display for tracking and debugging
    if debug != False:
        cv2.imshow('', frame)
        cv2.waitKey(1)
        pprint.pprint(camera2aruco_pose, width=1)

    return camera2aruco_pose, frameOut





# STANDALONE TEST
def standalone():
    # Video feed source
    cap = cv2.VideoCapture('https://192.168.0.103:8085/video')

    # Load camera calibration matrices
    with np.load('RedmiNote9Pro.npz') as X:
        mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]

    # Defining all the aruco-related stuff
    # marker map (implementation supports adding kalman Rv, but omitted here)
    markers = {
        0: ArucoMarker(0, [    0,    0,   0]),
        1: ArucoMarker(1, [ -640,    0,   0]),
        2: ArucoMarker(2, [-1100,  370,  -90]),
        }
    env_markers = [markers[i].id for i in markers.keys()]
    size_of_marker = 150
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()

    while 1:
        ret, frame = cap.read()
        corners, ids, rvecs, tvecs = detectMarkers(frame, aruco_dict, parameters, env_markers, size_of_marker, mtx, dist)

        if corners != None:
            camera2aruco_pose = getCamera2MarkerPose(ids, rvecs, tvecs, markers)

        cv2.imshow('', drawDetectedMarkers(frame, corners, ids, rvecs, tvecs, mtx, dist))
        cv2.waitKey(1)

        os.system('clear')
        pprint.pprint(camera2aruco_pose, width=1)

if __name__ == '__main__':
    standalone()
