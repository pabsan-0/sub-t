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


#########################
### STATE SPACE SETUP ###
#########################

# Load data and compute Y to later get the measurement noise covariance matrix
a = np.loadtxt('data1.txt')

x   = a[:,1]
z   = a[:,2]
yaw = a[:,3]

x_train   = x[0:1000]
z_train   = z[0:1000]
yaw_train = yaw[0:1000]

y = np.array([x_train, z_train, yaw_train], dtype=np.float32)

# System model

Ts = 0.0386

A = np.array([\
    [1, Ts,  0, 0,   0, 0],
    [0, 0,   0, 0,   0, 0],
    [0, 0,   1, Ts,  0, 0],
    [0, 0,   0, 0,   0, 0],
    [0, 0,   0, 0,   1, Ts],
    [0, 0,   0, 0,   0, 0]])

B = np.zeros([6,1]);

C = np.array([\
    [1, 0,   0, 0,   0, 0],
    [0, 0,   1, 0,   0, 0],
    [0, 0,   0, 0,   1, 0]])

Rw = np.array([\
    [1, 0,   0, 0,   0, 0],
    [0, 1,   0, 0,   0, 0],
    [0, 0,   1, 0,   0, 0],
    [0, 0,   0, 1,   0, 0],
    [0, 0,   0, 0,   1, 0],
    [0, 0,   0, 0,   0, 1]])

Rv = np.cov(y)





#########################
### CLASSES AND FUNCS ###
#########################


class KalmanFilter(object):
    '''
    Implements a Linear Kalman Filter that can be used online to filter a signal.
    Initialize with system model matrices.

    ARGS:
        A: From the propagation equation for State Space models.
        B: From the propagation equation for State Space models.
        C: From the measurement equation for State Space models.
        D: From the measurement equation for State Space models. NOT USED.
        Q: Process noise covariance matrix.
        R: Measurement noise covariance matrix.
    '''

    def __init__(self, A, B, C, D, Q, R):
        # System model: propagation
        self.A = A
        self.B = B
        self.Q = Q

        # System model: measure
        self.C = C
        self.D = D
        self.R = R

        # Get state and measure vector dimensions from matrices
        n_elements_x = A.shape[0]
        n_elements_y = C.shape[0]

        # Kalman gains
        self.L = np.zeros([n_elements_x, n_elements_y], dtype=np.float32)
        self.P = np.zeros([n_elements_x, n_elements_x], dtype=np.float32)

        # Q coefficient
        self.LL = np.eye(n_elements_x)

        # Initial state vector
        self.xhat = np.zeros([n_elements_x], dtype=np.float32)

        # store blueprints of scalable matrices
        self.C_blueprint = C

    def propagate(self, u):
        # Compute next state with propagation equation
        self.xhat = self.A @ self.xhat + self.B  @ u

        # Compute P
        self.P = self.A @ self.P @ self.A.T + self.LL @ self.Q @ self.LL.T

    def update(self, y):
        # Compute the Kalman gain
        self.L = self.P @ self.C.T @ inv(self.C @ self.P @ self.C.T + self.R)

        # Correct xhat from weighted sum of current xhat + error * kalman gain
        self.xhat = self.xhat + self.L @ (y - self.C @ self.xhat)

        # Adjust P
        self.P = self.P - self.L @ self.C @ self.P

    def filter_step(self, u, y, ret_xhat=False):
        self.propagate(u)
        self.update(y)
        if ret_xhat == True:
            return self.xhat
        else:
            return self.C @ self.xhat



class ArucoMarker(object):
    '''
    Implements a marker object to store its pose and kalman gain.

    ARGS:
        id:   ID of this ARUCO marker.
        pose: Provide as list [x, z, yaw]. Pose of this ARUCO marker.
        Rv:   Measurement noise covariance matrix of the camera seeing this code.
    '''

    def __init__(self, id, pose, Rv):
        # Store marker ID, pose
        self.id = id
        self.pose = pose

        # Store Kalman gain and Measurement Noise Covariance
        self.L = np.zeros([6, 3], dtype=np.float32)
        self.R = Rv

        # Compute and store the rotation matrix of this marker
        x = pose[0]
        z = pose[1]
        yaw = np.deg2rad(pose[2])
        self.rot = np.array([[np.cos(yaw), -np.sin(yaw),   x],\
                             [np.sin(yaw),  np.cos(yaw),   z],\
                             [          0,            0,   1]])

    def transform2D(self, pose_camera2marker):
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
        return A * np.array([1,1,0]) + yaw * np.array([0,0,1])


def get_camera_pose(markerID=0):
    '''
    Gets the camera pose with respect to any set of detected IDs.
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

        eul   = -cv2.decomposeProjectionMatrix(T[:3,:])[6]
        yaw   =  eul[1,0]
        pitch = (eul[0,0]+(90))*math.cos(eul[1,0])
        roll  = (-(90)-eul[0,0])*math.sin(eul[1,0]) + eul[2,0]

        # Add each marker's pose to the detection dictionary
        detections[idx] =  {#'raw': [int(markerID), x, y, z, yaw, roll, pitch],
                            'id': int(markerID),
                            'pose': np.array([x, z, yaw])}

    return detections






#####################
### PROGRAM START ###
#####################

# Create a library defining all the ARUCOs that exist in the environment
markers = {
    0: ArucoMarker(0, [    0,    0,   0], Rv),
    1: ArucoMarker(1, [ -640,    0,   0], Rv),
    2: ArucoMarker(2, [-1100,  370,  -90], Rv),
    }
env_markers = [markers[i].id for i in markers.keys()]

# define size of marker for scale, free units [currently in mm]
# using a common size saves some code work
size_of_marker = 150

# Instantiate a kalman filter and placeholder for its output
KF = KalmanFilter(A, B, C, [], Rw, Rv)
xhats = []

# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]

# Initialize the video recorder from IP camera
cap = cv2.VideoCapture('https://192.168.0.101:8085/video')



# Progress in time
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Work out the aruco markers from the picture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If some markers are found in the picture
    if corners != []:
        # If a marker that is not expected appears (noise), just skip to next loop
        if not all([i in env_markers for i in ids]):
            print('UNEXPECTED ARUCO MARKER SPOTTED, SKIPPING TO NEXT LOOP...')
            continue

        # get the rotation and traslation vectors CAMERA -> ARUCO
        rvecs,tvecs,trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)

        # Get camera pose. Contained because pose algebra may fail
        detections = get_camera_pose()

        # Terminal display 1
        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n Raw poses (x, z, yaw) w.r. to each marker')
        pprint.pprint(detections, width=1)

        # Draw aruco markers and 3D axes on picture
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        for i, j in zip(rvecs, tvecs):
            frame = aruco.drawAxis(frame, mtx, dist, i, j, 50)

        # markers that are being detected ATM
        markers_on_sight = [detections[i]['id'] for i in detections.keys()]

        # correct detection based on coordinates
        for idx, marker in enumerate(markers_on_sight):
            detections[idx]['pose'] = markers[marker].transform2D(detections[idx]['pose'])

        # Terminal display 2
        print('\n Transformed poses (x, z, yaw) w.r. to (0,0,0)')
        pprint.pprint(detections, width=1)




        # KALMAN FILTER ONLINE MODIFICATIONS

        # Assemble new kalman filter matrices: Kalman Gain and C matrix
        KF.L = np.hstack([markers[i].L for i in markers_on_sight])
        KF.C = np.vstack([KF.C_blueprint for i in markers_on_sight])

        # Assemble new kalman filter matrix: Measurement noise covariance matrix
        canvas =  np.zeros([len(markers_on_sight)*3, len(markers_on_sight)*3])
        for idx, id in enumerate(markers_on_sight):
            canvas[0+3*idx:3+3*idx, 0+3*idx:3+3*idx] = markers[id].R
        KF.R = canvas

        # assemble inputs and measurements
        u = np.array([0])
        y = np.hstack([detections[i]['pose'] for i,j in enumerate(markers_on_sight)])

        # Perform a filtering step
        xhat = KF.filter_step(u, y,  ret_xhat=True)
        xhats.append(xhat)

        # Retrieve the updated kalman gain of each marker to storage
        for idx, id in enumerate(markers_on_sight):
            markers[id].L = KF.L[:, 0+idx*3 :3+3*idx]

        # Terminal display 3
        with np.printoptions(precision=0, suppress=True):
            print('\n Kalman filtered pose \n', xhat)
            print(f'\n memory usage {psutil.virtual_memory().percent} %' )

    # Display the resulting frame, pyrDown so it fits the screen
    cv2.imshow('frame', (frame))

    # stop looping on Q press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hold the capture until a second keypress is made, then close
# cap.release()
# cv2.waitKey()
# cv2.destroyAllWindows()


# Plotting results
f2, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
ax1, ax2, ax3 = axes.flatten()
ax1.plot(xhats[:,0], label='filtered x');   ax1.legend(loc='upper right')
ax2.plot(xhats[:,2], label='filtered z');   ax2.legend(loc='upper right')
ax3.plot(xhats[:,4], label='filtered yaw'); ax3.legend(loc='upper right')
plt.show()
