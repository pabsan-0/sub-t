import cv2
import numpy as np

'''
This script creates bounding box from instance segmentation map labels and
stores them in txt files with the same name as the source.

This is a development source version for a single file. You might want to check
its adaptation to batches of data in a different script.

DISCONTINUED
'''


picname = '01_levine_rgb_1_rdb_bag_100318.png'
sample_image = cv2.imread(f'D:/Workspace_win/sub-t-master/preparing-data/realdata/experimental/output/{picname}')
cv2.imshow('original', sample_image)
img = sample_image

with open(f'./{picname}.txt', 'w') as file:
    for item in [1,2,3,4]:
        # Create a workspace background the size of the picture
        canvas = np.zeros(img.shape[:2]).astype(np.uint8)

        # Import this item's areas as white maps
        canvas[np.where((img==item).all(axis=2))] = 255

        # Do contour operation on the canvas to detect possible multiple items
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('1',canvas)

        # fill region inside contours and reiterate contouring to reduce noise
        cv2.drawContours(canvas, contours, -1, 255, -1)
        cv2.imshow(str(item), canvas)

        # cv2.waitKey()
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for c in contours:
            # Compute this contour's properties
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(canvas,(x,y),(x+w,y+h),140,2)

            xc_pu = (x + x + w)//2  /img.shape[1]
            yc_pu = (y + y + h)//2  /img.shape[0]
            w_pu  = w               /img.shape[1]
            h_pu  = h               /img.shape[0]

            # Write output to text file
            file.write(f'{item} {xc_pu} {yc_pu} {w_pu} {h_pu} \n')
        #cv2.imshow('processed '+ str(item), canvas)

cv2.waitKey()
#cv2.destroyAllWindows()
