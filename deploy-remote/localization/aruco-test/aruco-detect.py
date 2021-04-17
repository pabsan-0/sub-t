import numpy as np
import cv2
import cv2.aruco as aruco

'''
REVISION PENDING
Use this script to receive live feed and find aruco patterns on it. Requires
a callibrated camera matrix and distance vector, which are for now hardcoded.
'''

cap = cv2.VideoCapture('https://130.240.152.20:8085/video')

# calibration pending!!
with np.load('iPhoneCam.npz') as X:
    mtx,dist,_,_ = [X[i] for i in('mtx','dist','rvecs','tvecs')]

mtx = np.array([[1.84021751e+03, 0.00000000e+00, 5.42779058e+02],
       [0.00000000e+00, 2.39630743e+03, 9.54928911e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-0.04838364,  0.25314362,  0.00389218, -0.04914811, -0.14082424]])



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    gray = aruco.drawDetectedMarkers(gray, corners)

    if corners != []:
        size_of_marker =  1 # side lenght of the marker in meter
        rvecs,tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker , mtx, dist)

        print('\n Corners \n', corners)
        print('\n Rvecs \n', rvecs)
        print('\n Tvecs \n', tvecs)
        print('\n Mtx \n', mtx)
        print('\n Dist \n', dist)

        for i, j in zip(rvecs, tvecs):
            gray = aruco.drawAxis(gray, mtx, dist, i, j, 2)


    # print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame',cv2.pyrDown(gray))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
