import time
import cv2.aruco as aruco
import numpy as np
import cv2

'''
REVISION PENDING

This script is used to record a video of a calibration pattern, that is also
generated within (and you must print), so that the aruco callibration function
will output the camera matrix and distance vector.
'''


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(3,3,.025,.0125,dictionary)
img = board.draw((200*3,200*3))

#Dump the calibration board to a file
cv2.imwrite('charuco-calibration-pattern.png',img)

# Quit to just print out the example file for calibration
# quit()

#Start capturing images for calibration
cap = cv2.VideoCapture('https://192.168.0.107:8085/video')

allCorners = []
allIds = []
decimator = 0
for i in range(500):

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)

    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])

    cv2.imshow('frame',cv2.pyrDown(gray))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator+=1

imsize = gray.shape

#Calibration fails for lots of reasons. Release the video if we do
try:
    print('Callibration started!')
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print('Saving calibration results to file...', end ='')
    np.savez('RedmiNote9Pro', retval=retval, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvecs=rvecs, tvecs=tvecs)
    print(' Saved!')
    print(cameraMatrix)
    print(distCoeffs)
except:
    print('FAILED CALIBRATION!')
    cap.release()


cap.release()
cv2.destroyAllWindows()
