import cv2
import numpy as np
import sys
import os


'''
This script creates bounding box from instance segmentation map labels and
stores them in txt files with the same name as the source. Working paths must
be HARDCODED at the begginning of this script.

It will also capture a record file that will hold how many items of which are there.
Use this file to diagnose noisy annotations that can be taken from image segmentation
maps that are not convex enough, leading to an overestimated number of items.
'''

# Define the path where the image labels are stored & where to dump the txts
# label_dir = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/experimental/'
label_dir = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PST900_RGBT_Dataset_900/test/labels/'
image_dir = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PST900_RGBT_Dataset_900/test/rgb/'
dump_dir  = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PST900_resized/images/'
outlabel_dir  = None

log_file  = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PST900_resized/test_dataset_summary.csv'



## These dicts just for big brain time
PSTRGB_ids = {
    'backpack':     2,
    'drill':        3,
    'extinguisher': 1,
    'survivor':     4,
}
goal_ids = {
    'backpack':     0,
    'drill':        2,
    'extinguisher': 3,
    'survivor':     4,
}

# this list to make sure the output classes ID matches goal_ids
pstrgb_2_goal_ids = [3, 0, 2, 4]


def resize_one_then_centercrop(image):
    target_h = 680
    target_w = 680

    # load & dims
    pic = image
    current_h = pic.shape[0]
    current_w = pic.shape[1]

    if current_h < current_w:
        # match pic_height
        scaleFactor = target_h / current_h
        pic = cv2.resize(pic, None, fx = scaleFactor, fy = scaleFactor, interpolation=cv2.INTER_NEAREST)

        # update dimensions
        current_h = pic.shape[0]
        current_w = pic.shape[1]

        # centercrop weight to desired size
        left_bound  = current_w // 2 - target_w //2
        right_bound = current_w // 2 + target_w //2
        pic = pic[:, left_bound:right_bound]

    else:
        # match pic_width
        scaleFactor = target_w / current_w
        pic = cv2.resize(pic, None, fx = scaleFactor, fy = scaleFactor, interpolation=cv2.INTER_NEAREST)

        # update dimensions
        current_h = pic.shape[0]
        current_w = pic.shape[1]

        # centercrop weight to desired size
        top_bound = current_h // 2 - target_h //2
        bot_bound = current_h // 2 + target_h //2
        pic = pic[top_bound:bot_bound, :]

    return pic



# Get names from label dir and store a list with all png names
pic_list = [file for file in os.listdir(label_dir) if file.split('.')[-1]=='png']

totalnum = len(pic_list)
tracker = 0

with open(log_file,'w') as logfile:
    # Write headers to file
    logfile.write('file, extinguisher, backpack, drill, survivor\n')
    logfile.write(',=sum(B3:B8000),=sum(C3:C8000),=sum(D3:D8000),=sum(E3:E8000)\n')

    # for each png name run annotation func
    for picname in pic_list:

        # print progress...
        print(f'\r{tracker} / {totalnum}', end ='')
        tracker += 1

        # reset the number of items that have appeared
        occurrences = [0, 0, 0, 0]

        # Read the LABEL & image picture
        label = cv2.imread(label_dir + picname)
        image = cv2.imread(image_dir + picname)

        # resize together to match sizes, reduce largest to 640 and pad with zeros
        label = resize_one_then_centercrop(label)
        image = resize_one_then_centercrop(image)

        # export resized imagepicutre & keep labels only if desired
        cv2.imwrite(dump_dir + picname, image)
        if outlabel_dir != None:
            cv2.imwrite(outlabel_dir + picname, label)

        # Initialize a textfile in the dump dir with the same name as the source picture
        textfilename = dump_dir + picname[:-4] + '.txt'
        with open(textfilename, 'w') as file:

            # Check for each possible item's mask
            for item in [1,2,3,4]:
                # Create a workspace background the size of the picture
                canvas = np.zeros(label.shape[:2]).astype(np.uint8)

                # Import this item's areas as a white map to the canvas
                canvas[np.where((label==item).all(axis=2))] = 255

                # Do contour operation on the canvas to detect possible multiple items
                contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # draw thick + fill contours and reiterate contouring to reduce noise
                cv2.drawContours(canvas, contours, -1, 255,  30)
                cv2.drawContours(canvas, contours, -1, 255, -1)
                contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                for c in contours:
                    # Compute this contour's properties
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(canvas,(x,y),(x+w,y+h),140,2)

                    # Compute per unit values in accordance to YOLO format
                    xc_pu = (x + x + w)//2  /label.shape[1]
                    yc_pu = (y + y + h)//2  /label.shape[0]
                    w_pu  = w               /label.shape[1]
                    h_pu  = h               /label.shape[0]

                    # Write output to text file
                    file.write(f'{pstrgb_2_goal_ids[item-1]} {xc_pu} {yc_pu} {w_pu} {h_pu}\n')
                    occurrences[item-1] += 1
        some_string = ",".join([str(i) for i in occurrences])
        logfile.write( f'{picname},{some_string}\n')
