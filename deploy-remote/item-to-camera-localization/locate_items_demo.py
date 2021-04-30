import cv2
import numpy as np

'''
This DEMONSTRATION script is used to get real-world coordinates of items with
respect to the camera coordinate frame, in the horizontal plane.

For this, it is neccesary to know the real world distance between two points
MEASURED IN A PLANE PERPENDICULAR TO THE CAMERA AXIS of this item i.e. its width.

The function will ask for three values:
    The horizontal image coordinates of a first point (left side)
    The horizontal image coordinates of a second point (right side)
    The real world distance between these two points

To simplify the model, the image points are measured on an image which is already
distortion-corrected by using the calibration parameters of the camera.
'''

# Load camera calibration matrices
with np.load('RedmiNote9Pro.npz') as X:
    mtx, dist = [X[i] for i in('cameraMatrix', 'distCoeffs')]
print('\n Camera intrinsic matrix: \n',mtx)
print('\n Distortion coefficients: \n', dist)

# Get video feed from camera
cap = cv2.VideoCapture('http://192.168.0.102:8085/video')

while 1:
    ret, frame = cap.read()

    # Undistort image with correction parameters from calibration
    dst = cv2.undistort(frame, mtx, dist, None)

    # Show both vanilla and undistorted image
    cv2.imshow('original', frame)
    cv2.imshow('undistorted', dst)

    print('Measure the two points horizontal coordinates and press Q to continue.')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# get inputs
u1 = int(input('Select X image coordinate of point 1, u1 (left side): '))
u2 = int(input('Select X image coordinate of point 2, u2 (right side): '))
dx = int(input('Select Real world horizontal distance between both points: '))

# Perform math to get results (see README)
fx = mtx[0,0]
cx = mtx[0,2]
s = fx * (dx) / (u2 - u1)
A = (u1 - cx)/(u2 - cx)
x1 = (A * dx) / (1 - A)
x2 = x1 + dx
z = (fx * x1 - s * u1) / cx


# Printout real world coordinates
print()
print('Computed distance ', z)
print('Computed X1 ', x1)
print('Computed X2 ', x2)

cv2.waitKey()
cv2.destroyAllWindows()
