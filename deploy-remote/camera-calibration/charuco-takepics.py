import numpy as np
import cv2
import cv2.aruco as aruco
from sys import argv
import os

'''
Use this script to receive live feed and take pictures that will serve for
calibration.
'''

# load ip from args 'https://192.168.0.107:8085/video'
try:
    ipdir = argv[1]
except:
    pritn('Trouble with your argument, loading IP from script default!')
    ipdir = 'https://192.168.0.106:8085/video'

# initialize the video recorder from IP camera and display init message. Init counter
cap = cv2.VideoCapture(ipdir)
print('Initialized! Press "q" to take a picture. Focus terminal and press Ctrl+C to exit at any point.')
z=0

# clear or create dump folder
os.system('rm -r ./calib-pics')
os.system('mkdir ./calib-pics')

# Record image until stopped with Ctrl C
while(True):
    # Capture current frame and create a copy for resized display
    ret, frame = cap.read()
    displayed_frame = frame.copy()

    # optionally pyrDown so video fits the screen
    for i in range(1): #-----------------------------------------------# MODIFY
        displayed_frame = cv2.pyrDown(frame)

    # display current video
    cv2.imshow('frame', displayed_frame)

    # wait for Q keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):

        # save current frame to file
        cv2.imwrite(f'./calib-pics/charuco-calib-{z}.png', frame)
        print(f'Saved as "charuco-calib-{z}.png"!')
        z += 1

# Hold the capture until a second keypress is made, then close
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
