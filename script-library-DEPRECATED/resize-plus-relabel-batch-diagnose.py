import cv2
import numpy as np

'''
This script checks the after-resize label maps and images to verify that the
noisy image segmentation annotations are not messing up the bounding boxes.
Use it to modify parameters and then redo the modifications on the batch resizer.
'''


picname = '03_levine_rgb_3_rdb_bag_300637.png'
image = cv2.imread(f'D:/Workspace_win/sub-t-master/preparing-data/realdata/experimental/output/{picname}')
label = cv2.imread(f'D:/Workspace_win/sub-t-master/preparing-data/realdata/experimental/out_label/{picname}')

cv2.imshow('original', image)


for item in [1,2,3,4]:
    # Create a workspace background the size of the picture
    canvas = np.zeros(label.shape[:2]).astype(np.uint8)

    # Import this item's areas as white maps
    canvas[np.where((label==item).all(axis=2))] = 255

    # Do contour operation on the canvas to detect possible multiple items
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw thick + fill contours and reiterate contouring to reduce noise
    cv2.drawContours(canvas, contours, -1, 255,  30)
    cv2.drawContours(canvas, contours, -1, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('3 - ' + str(item), canvas)

    for c in contours:
        # Compute this contour's properties
        x,y,w,h = cv2.boundingRect(c)
        image = cv2.rectangle(image,(x,y),(x+w,y+h),140,3)

cv2.imshow('final' + str(item), image)


cv2.waitKey()
#cv2.destroyAllWindows()
