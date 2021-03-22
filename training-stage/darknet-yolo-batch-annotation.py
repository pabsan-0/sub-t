import cv2
import colorsys
import os

'''
This script is meant to read a yolo log file with batch inferred results
and draw predicted bounding boxes on the source pictures. Run it with
hardcoded paths that you NEED TO MODIFY, READ THE SCRIPT THROUGH.

The file needed as input can be obtained by running this command:
./darknet detector test PATH/obj.data PATH/yolov4.cfg PATH/yolov4.weights -dont_show -ext_output < PATH/test.txt > results_yo.txt

Where test.txt is a textfile with the absolute path of each of the images
to be batch-inferred and results_yo.txt is the output that will be generated
by darknet. Inside this file, each predicted image should look something 
like this, adjust file scraping to if neccesary for further releases:

"""
Enter Image Path:  Detection layer: 30 - type = 28 
 Detection layer: 37 - type = 28 
/home/pablo/YOLOv4/PPU-6/test/25_exyn_bag15_rgb_frame1500305.png: Predicted in 104.754000 milli-seconds.
drill: 66%	(left_x:  409   top_y:  342   width:   41   height:   58)
survivor: 99%	(left_x:  562   top_y:  298   width:  120   height:  269)
backpack: 30%	(left_x:  650   top_y:  490   width:   33   height:   61)
Enter Image Path:  Detection layer: 30 - type = 28 
 Detection layer: 37 - type = 28 
"""
 
'''

# define paths
out_dir = '/home/pablo/YOLOv4/infer_output/'
text_file_path =  '/home/pablo/YOLOv4/darknet/results_yo.txt'

# create ouput path if doesnt exist + init counter 'idx'
os.mkdir(out_dir)
idx = 0

# required format params for drawing annotations
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5

# read the output of the YOLO log to load predictions
with open(text_file_path, 'r') as file:
    for line in file.readlines():

        # when a new picture is introduced, save previous and load+resize new
        if '/home/pablo/' in line:
            # save current picture to file unless first iteration
            if idx != 0:
                cv2.imwrite(f'{out_dir + str(idx)}.jpg', pic)
                print(f'Saved at {out_dir + str(idx)}.jpg')
                
            # load file as cv2 image from text file
            picpath = line.split(':')[0]
            pic = cv2.imread(picpath)
            idx += 1
            
            # resize to small size & store original size to work labels later
            picsize_original = pic.shape[0]
            pic = cv2.resize(pic, (680,680))
            
        
        # when a prediction is recorded, add the bbox to the current image
        if '%' in line:
            # read the class + confidence score as title
            title = line.split('%')[0] + '%'
            
            # use the class initial to define color
            hue = ord(title[0]) - (100.5 - ord(title[0])) * 15
            color = tuple(255 * i  for i in colorsys.hsv_to_rgb(hue/360.0, 1, 1))
            
            # work out where to draw the bounding box depending on output img size
            scalefactor = 680 / picsize_original
            x1, y1, w, h = [int(abs(int(i))*scalefactor) for i in line[:-2].replace('-','').split() if i.isdigit()]
            
            # actually overwrite the picture with box + title
            pic = cv2.rectangle(pic, (x1, y1), (x1 + w, y1 + h), color, thickness)
            pic = cv2.putText(pic, title, (x1, y1), font, fontScale, color, thickness, cv2.LINE_AA)
