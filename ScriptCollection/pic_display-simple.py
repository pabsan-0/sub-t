import cv2
import os
import sys


def display(path):
    """ Displays a picture plotting bboxes read from a .txt file.
    Path must be a string with a picture name WITHOUT extension
    In the target dir, a .jpeg & .txt files with the Path name must exist.
    """
    cat_dic = {
        0: 'backpack',
        1: 'survivor',
        2: 'cell_phone',
        3: 'fire_extinguisher',
        4: 'drill',
        5: 'helmet',}

    pic = cv2.imread(f'{path}.jpeg')
    color = (255, 0, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    with open(f'{path}.txt') as file:
        for i in file.readlines():
            cat, x, y, w, h = [float(k) for k in i.split()]

            x1, y1 = int(x * pic.shape[0]), int(y * pic.shape[0])
            x2, y2 = int(x1 + w * pic.shape[0]), int(y1 + h * pic.shape[0])

            pic = cv2.rectangle(pic, (x1, y1), (x2, y2), color, thickness)
            pic = cv2.putText(pic, cat_dic[int(cat)], (x1, y1), font,
                              fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('a', pic)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = sys.argv[1]
    display(path)
