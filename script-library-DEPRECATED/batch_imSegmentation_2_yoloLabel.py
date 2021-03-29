import cv2
import numpy as np
import sys
import os

''' Pablo Santana - 01/03/2021

This script creates bounding box from instance segmentation map labels and
stores them in txt files with the same name as the source. Working paths must
be HARDCODED at the MAIN of this script.

This is the final version of this tool and is used for converting batches of
images.

DISCONTINUED
'''

def annotate(picname, rgb_dir):

    # Read the LABEL picture
    img = cv2.imread(picname)

    # Initialize a textfile in the dump dir with the same name as the source picture
    textfilename = rgb_dir + '/' + picname[:-4] + '.txt'

    with open(textfilename, 'w') as file:

        # Check for each possible item's mask
        for item in [1,2,3,4]:
            # Create a workspace background the size of the picture
            canvas = np.zeros(img.shape[:2]).astype(np.uint8)

            # Import this item's areas as a white map to the canvas
            canvas[np.where((img==item).all(axis=2))] = 255

            # Do contour operation on the canvas to detect possible multiple items
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                # Compute this contour's properties
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(canvas,(x,y),(x+w,y+h),140,2)

                # Compute per unit values in accordance to YOLO format
                xc_pu = (x + x + w)//2  /img.shape[1]
                yc_pu = (y + y + h)//2  /img.shape[0]
                w_pu  = w               /img.shape[1]
                h_pu  = h               /img.shape[0]

                # Write output to text file
                file.write(f'{item} {xc_pu} {yc_pu} {w_pu} {h_pu}\n')

if __name__ == '__main__':

    # Define the path where the image labels are stored & where to dump the txts
    label_dir = 'D:/Workspace_win/sub-t-master/realdata/PSTRGB_Dataset/labels'
    rgb_dir = 'D:/Workspace_win/sub-t-master/realdata/PSTRGB_Dataset/rgb'

    # These for debug
    # label_dir = 'D:/Workspace_win/sub-t-master/realdata/experimental'
    # rgb_dir = 'D:/Workspace_win/sub-t-master/realdata/experimental'

    # Place cwd in label dir and store a list with all png names
    os.chdir(label_dir)
    pic_list = [file for file in os.listdir() if file.split('.')[-1]=='png']

    # for each png name run annotation func
    for picname in pic_list:
        annotate(picname, rgb_dir)
