################################################################################
#    main-multiprocessing.py / pabloItem2CameraPosition                        #
#    Pablo Santana - CNNs for object detection in subterranean environments    #
################################################################################
# p0 (main)         p1                   p2                      p5
# ┌────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
# │ImageCapture│ q1 │Estimate Camera│ q3 │Kalman-Filter    │q4  │Draw Worldmap │
# │&& Undistort├┬──>│Pose From ArUco├───>│Camera Pose      ├──┬>│with cam FoV &│
# │            ││   │markers        │    │                 │  │ │item positions│
# └────────────┘│   └───────────────┘    └─────────────────┘  │ └──────────────┘
#               │   p3                   p4                   │
#               │   ┌────────────────┐   ┌─────────────────┐  │
#               │q2 │Object detection│q5 │Estimate item    │q6│
#               └──>│with Darknet    ├──>│position w.r to  ├──┘
#                   │(CNN)           │   │the camera       │
#                   └────────────────┘   └─────────────────┘
#                                              └─YOU'RE HERE

import numpy as np
import cv2

def filterDetections(detections, itemListAndSize):
    ''' Filter the detections: only items of interest + horizontal coordinates
    '''
    filteredComplete = [i for i in detections if i[0] in itemListAndSize]
    filteredPosition = [(i[0], i[2][0], i[2][2]) for i in detections if i[0] in itemListAndSize]
    return filteredComplete, filteredPosition


def getItemPosition(frameDetected, originalImage, itemDetections, itemListAndSize, mtx):
    ''' Computes the horizontal position of an item's bbox w.r. to the camera.
    '''
    # filter detections so only items of interest are fed forward
    _, filteredPosition = filterDetections(itemDetections, itemListAndSize)

    # Adjust the scaling factor, input_width=800 but darknet_width=416 (dets)
    rescaling_factor = originalImage.shape[1] / frameDetected.shape[1]

    item2CameraPosition = []
    for (bbox_class, bbox_hpos, bbox_width) in filteredPosition:
        # compute the pixel position in full-size image
        u1 = (bbox_hpos - bbox_width /2) * rescaling_factor
        u2 = (bbox_hpos + bbox_width /2) * rescaling_factor
        dx = itemListAndSize[bbox_class]

        # Perform math to get results (see README)
        fx = mtx[0,0]
        cx = mtx[0,2]
        s = fx * (dx) / (u2 - u1)
        A = (u1 - cx)/(u2 - cx)
        x1 = (A * dx) / (1 - A)
        x2 = x1 + dx
        z = (fx * x1 - s * u1) / cx

        # Store as part of many possible positions
        item2CameraPosition.append([bbox_class, z, (x1+x2)/2])

    return item2CameraPosition
