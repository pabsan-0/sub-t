################################################################################
#    main-multiprocessing.py / pabloKalman                                     #
#    Pablo Santana - CNNs for object detection in subterranean environments    #
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │yaw EUCLIDEAN    │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#               │   p3                   p4    └─YOU'RE HERE  │
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#

import numpy as np
from numpy.linalg import inv
import os

def angdiff(b1, b2):
	r = (b2 - b1) % np.pi
	if r >= np.pi:
		r -= 2 * np.pi
	return r


def modelImport(sampleTime=0.0386, dataSample=[]):
    # System model

    Ts = sampleTime

    A = np.eye(4)

    B = np.zeros([4,1]);

    C = np.eye(4)

    # lower -> more inertia
    Rw = np.array([\
        [0.05,    0,    0,    0],
        [0,    0.05,    0,    0],
        [0,      0,  0.1,    0],
        [0,      0,    0,  0.1]])

    if dataSample != []:
        data = np.loadtxt(dataSample)
        x   = data[:,1]
        z   = data[:,2]
        yaw = data[:,3]
        y = np.array([x, z, yaw], dtype=np.float32)
        Rv = np.cov(y)
    else:
        Rv = np.eye(3)

    Rv = np.array([\
        [10,    0,    0,    0],
        [0,    10,    0,    0],
        [0,    0,   10,    0],
        [0,    0,    0,   10]]) * 1

    return A, B, C, None, Rw, Rv


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



# CALL FROM LOOP
def main(KF, camera2aruco_pose, markers):
    # KALMAN FILTER ONLINE MODIFICATIONS

    # infer which markers are being seen
    markers_on_sight = [camera2aruco_pose[i]['id'] for i in camera2aruco_pose.keys()]

    # Assemble new kalman filter matrices: Kalman Gain and C matrix
    KF.L = np.hstack([markers[i].L for i in markers_on_sight])
    KF.C = np.vstack([KF.C_blueprint for i in markers_on_sight])

    # Assemble new kalman filter matrix: Measurement noise covariance matrix
    canvas =  np.zeros([len(markers_on_sight)*4, len(markers_on_sight)*4])

    for idx, id in enumerate(markers_on_sight):
        canvas[0+4*idx:4+4*idx, 0+4*idx:4+4*idx] = markers[id].R
    KF.R = canvas


    ''' DOESNT WORK - KEPT FOR REFERENCE
    # Linearize angles with respect to current average
    # Compute average from all measurements
    angle_list = [np.deg2rad(camera2aruco_pose[i]['pose2world'][2]) for i,j in enumerate(markers_on_sight)]
    cos_avg = np.average([np.cos(i) for i in angle_list])
    sin_avg = np.average([np.sin(i) for i in angle_list])
    angle_avg = np.arctan2(sin_avg, cos_avg)
    #
    # compute the angle mean and subtract it from each measurement
    for i, j in enumerate(markers_on_sight):
        camera2aruco_pose[i]['pose2world'][2] = np.rad2deg(angle_list[i] - angle_avg)
    '''

    # convert angle to cos-sin and pack it up into camera2aruco_pose dict structure
    for i, j in enumerate(markers_on_sight):
        camera2aruco_pose[i]['pose2world'] = np.append(camera2aruco_pose[i]['pose2world'], 0)
        camera2aruco_pose[i]['pose2world'][3] = np.sin(np.deg2rad(camera2aruco_pose[i]['pose2world'][2]))
        camera2aruco_pose[i]['pose2world'][2] = np.cos(np.deg2rad(camera2aruco_pose[i]['pose2world'][2]))


    # assemble inputs and measurements
    u = np.array([0])
    y = np.hstack([camera2aruco_pose[i]['pose2world'] for i,j in enumerate(markers_on_sight)])

    # Perform a filtering step
    xhat = KF.filter_step(u, y, ret_xhat=True)

    ''' DOESNT WORK - KEPT FOR REFERENCE
    # undo the linear transformation of the angle and convert to +180/-180 space
    xhat[4] = np.mod(xhat[4] + np.rad2deg(angle_avg) - 180, 360) - 180
    '''

    # Retrieve the updated kalman gain of each marker to storage
    for idx, id in enumerate(markers_on_sight):
        markers[id].L = KF.L[:, 0+idx*3 :3+3*idx]

    return xhat[0], xhat[1], np.rad2deg(np.arctan2(xhat[3], xhat[2]))


if __name__ == '__main__':
    print('You cannot run this script standalone. Go to main.py and call from there.')
