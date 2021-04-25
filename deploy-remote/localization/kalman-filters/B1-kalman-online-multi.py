import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

a = np.loadtxt('data1.txt')

x   = a[:,1]
z   = a[:,2]
yaw = a[:,3]

x_train   = x[0:1000]
z_train   = z[0:1000]
yaw_train = yaw[0:1000]

x_valid   = x[1001:]
z_valid   = z[1001:]
yaw_valid = yaw[1001:]


# Data overview
f1, axes = plt.subplots(3, 2, figsize=(15, 6))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
ax1.plot(x_train); ax1.legend('X', loc='upper right')
ax2.hist(x_train, 100); ax2.legend('X', loc='upper right')
ax3.plot(z_train); ax3.legend('Z', loc='upper right')
ax4.hist(z_train, 100); ax4.legend('Z', loc='upper right')
ax5.plot(yaw_train); ax5.legend('Yaw', loc='upper right')
ax6.hist(yaw_train, 100); ax6.legend('Yaw', loc='upper right')


# System model
u = np.zeros([1, len(x_train)], dtype=np.float32)
y = np.array([x_train, z_train, yaw_train], dtype=np.float32)

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

    EXAMPLE:
        # import required libs
        import numpy as np
        from numpy.linalg import inv

        # Initialize Kalman filter & placeholder
        kf = KalmanFilter(A, B, C, D, Q, R)
        yhat_list = []

        while 1:
            # Obtain measurements from sensors/disk
            current_u = f(. . .)
            current_y = g(. . .)

            # (Optional) redefine matrices in state space if sample time changes
            # ts = 0.05
            # kf.A = np.array([[1, ts],[0, 1]])

            # Filter the received measurements to get an estimate
            yhat = KalmanFilter.filter_step(current_u, current_y)

            # Store the estimate in a variable for history
            yhat_list.append(yhat)
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
        self.id = id
        self.pose = pose
        self.L = np.zeros([6, 3], dtype=np.float32)
        self.R = Rv



# Create a library defining all the ARUCOs that exist in the environment
markers = {
    0: ArucoMarker(0, [0,   0,  0], Rv),
    1: ArucoMarker(1, [0,  15,  0], Rv),
    2: ArucoMarker(2, [20, 15, 90], Rv),
    }



# Instantiate a kalman filter and placeholder for its output
KF = KalmanFilter(A, B, C, [], Rw, Rv)
yhats = []

# Simulating a 2-markers situation
y = np.array([x_train, z_train, yaw_train, x_train, z_train, yaw_train], dtype=np.float32)

# progress in time
for i in range(max(u.shape)-1):

    # markers that are being detected ATM
    markers_on_sight = [0,1]

    # correct detection based on coordinates
    # y = y

    # Assemble new kalman filter matrices: Kalman Gain and C matrix
    KF.L = np.hstack([markers[i].L for i in markers_on_sight])
    KF.C = np.vstack([KF.C_blueprint for i in markers_on_sight])

    # Assemble new kalman filter matrix: Measurement noise covariance matrix
    canvas =  np.zeros([len(markers_on_sight)*3, len(markers_on_sight)*3])
    for idx, id in enumerate(markers_on_sight):
        canvas[0+3*idx:3+3*idx, 0+3*idx:3+3*idx] = markers[id].R
    KF.R = canvas

    # Perform a filtering step
    yhat = KF.filter_step(u[:,i], y[:,i])
    yhats.append(yhat)

    # Retrieve the updated kalman gain of each marker to storage
    for idx, id in enumerate(markers_on_sight):
        markers[id].L = KF.L[:, 0+idx*3 :3+3*idx]



# Plotting results
yhat = np.array(yhats)
f2, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
ax1, ax2, ax3 = axes.flatten()
ax1.plot(x_train, label='x');     ax1.plot(yhat[:,0], label='filtered x');   ax1.legend(loc='upper right')
ax2.plot(z_train, label='z');     ax2.plot(yhat[:,1], label='filtered z');   ax2.legend(loc='upper right')
ax3.plot(yaw_train, label='yaw'); ax3.plot(yhat[:,2], label='filtered yaw'); ax3.legend(loc='upper right')
plt.show()