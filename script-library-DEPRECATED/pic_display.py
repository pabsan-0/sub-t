import cv2
import os
import sys

'''
This script displays the bounding boxes contained in a YOLO-annotated picture.
Modify this script to fix picture format, classes... etc.

TRUSTED: CHECKED WITH OUTPUT FROM YOLO_MARK
See https://github.com/AlexeyAB/Yolo_mark for more information.



DEMOS:

cd D:\Workspace_win\sub-t-master

# synth data unity
python .\pic_display.py .\synthdata\renderings_unity\json2yolo\test\rgb_2

# real data
python .\pic_display.py .\realdata\PSTRGB_Dataset\rgb\01_levine_rgb_1_rdb_bag_100400
'''

cat_dic = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    }

def display(path):
    """ Displays a picture plotting bboxes read from a .txt file.
    Path must be a string with a picture name WITHOUT extension
    In the target dir, a .jpeg & .txt files with the Path name must exist.
    """
    pic = cv2.imread(f'{path}.png')
    color = (255, 0, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    with open(f'{path}.txt') as file:
        for i in file.readlines():
            cat, x, y, w, h = [float(k) for k in i.split()]

            x1, y1 = int((x - w/2) * pic.shape[1]),  int((y - h/2) * pic.shape[0])
            x2, y2 = int(x1 + w * pic.shape[1]), int(y1 + h * pic.shape[0])

            pic = cv2.rectangle(pic, (x1, y1), (x2, y2), color, thickness)

            # Comment the 2 following lines for class-agnostic boxes
            pic = cv2.putText(pic, cat_dic[int(cat)], (x1, y1), font,
                          fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('a', pic)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = sys.argv[1]
    display(path)
