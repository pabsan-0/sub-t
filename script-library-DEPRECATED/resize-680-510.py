import cv2
import numpy as np
import os
'''
CALL THIS FUNC FROM sub-t-master/realdata/RopeArtifact
'''

target_h = 510
target_w = 680

for i, picname in enumerate(os.listdir('./RopePicsRaw/')):

    # load & dims
    pic = cv2.imread('./RopePicsRaw/' + picname)
    current_h = pic.shape[0]
    current_w = pic.shape[1]

    # match pic_height
    scaleFactor = target_h / current_h
    pic = cv2.resize(pic, None, fx = scaleFactor, fy = scaleFactor)

    # update dimensions
    current_h = pic.shape[0]
    current_w = pic.shape[1]

    # centercrop weight to desired size
    left_bound  = current_w // 2 - target_w //2
    right_bound = current_w // 2 + target_w //2
    pic = pic[:, left_bound:right_bound]

    # update dimensions
    current_h = pic.shape[0]
    current_w = pic.shape[1]

    # filter just in case
    if (current_h == target_h) & (current_w == target_w):
        cv2.imwrite(f'./resized_output/ropeArtifact-{i}.png', pic)
