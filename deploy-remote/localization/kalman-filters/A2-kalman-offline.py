import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

'''
Implements a Kalman filter to be run on recorded data (offline)
Implemented in python from previous version
'''

def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.array(*args, **kwargs)


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


# system model

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



# FAILS, OVERFLOWS AND BREAKS

def custom_tv_kalman(u, y, A, B, C, D, Q, R, LL, L0, P0):
    ### Time variant Kalman filter
    # Q: Covariance matrix for process noise
    # R: Covariance matrix for measurement noise
    # L0: initial filter gain value - initiated within the function
    # P0: initial covariance estimation error - initiated within the function

    xhat = np.zeros([6,max(u.shape)], dtype=np.float32)
    yhat = np.zeros([3,max(u.shape)], dtype=np.float32)

    L = np.zeros([6,3,max(u.shape)], dtype=np.float32)  # filter gain K
    P = np.zeros([6,6,max(u.shape)], dtype=np.float32)  # covariance estimation error

    # we are at instant K (known state) aiming to predict k+1
    for k in range(max(u.shape)-1):
        # Propagation loop
        xhat[:,k+1] = A @xhat[:,k]       + B @ u[:,k]
        P[:,:,k+1]  = A @ P[:,:,k] @ A.T + LL @ Q @ LL.T

        # Upgrade loop
        L[:,:,k+1]  = P[:,:,k+1] @ C.T @ inv(C@P[:,:,k+1]@C.T + R)
        xhat[:,k+1] = xhat[:,k+1] + L[:,:,k+1] @ (y[:,k+1] - C @ xhat[:,k+1])
        P[:,:,k+1]  = P[:,:,k+1] - L[:,:,k+1] @ C@P[:,:,k+1]

        yhat[:,k+1] = C @ xhat[:,k+1]

    return yhat.T


yhat = custom_tv_kalman(u, y, A, B, C, [], Rw, Rv, np.eye(6), [], [])

# Results overview
f2, axes = plt.subplots(3, 1, figsize=(15, 6))
ax1, ax2, ax3 = axes.flatten()
ax1.plot(x_train, label='x');     ax1.plot(yhat[:,0], label='filtered x');   ax1.legend(loc='upper right')
ax2.plot(z_train, label='z');     ax2.plot(yhat[:,1], label='filtered z');   ax2.legend(loc='upper right')
ax3.plot(yaw_train, label='yaw'); ax3.plot(yhat[:,2], label='filtered yaw'); ax3.legend(loc='upper right')


plt.show()
