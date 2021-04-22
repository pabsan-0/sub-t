import time
import cv2.aruco as aruco
import numpy as np
import cv2
import os

'''
This script is used to calibrate a camera from a set of pictures of a CHARUCO
pattern. Said charuco can be produced with this script.
'''

# load the aruco dictionary that is going to be used
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(3,3,.025,.0125,dictionary)
img = board.draw((200*3,200*3))

#Dump the calibration board to a file
cv2.imwrite('charuco-calibration-pattern.png',img)
# quit()


# Move to folder where pictures are and iterate them
os.chdir('./calib-pics-charuco')

allCorners = []
allIds = []
decimator = 0
for i in os.listdir('./'):
    print('Loaded ', i)
    frame = cv2.imread(i)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)

    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
            allCorners.append(res2[1])
            allIds.append(res2[2])
        else:
            print(f'issue with picture {i}')
imsize = gray.shape

#Calibration fails for lots of reasons. Release the video if we do
os.chdir('..')
try:
    print('Callibration started!')
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print('Saving calibration results to file...', end ='')
    np.savez('RedmiNote9Pro', retval=retval, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvecs=rvecs, tvecs=tvecs)
    print(' Saved!')
    print(cameraMatrix)
    print(distCoeffs)
except:
    print('FAILED CALIBRATION! Please retry')

cv2.destroyAllWindows()
