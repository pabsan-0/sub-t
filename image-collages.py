import os
import cv2
import numpy as np
from sys import argv

'''
This script concatenates all images in a directory (provided as argument) 
so that the output is a high-resolution collage of them. 

For concatenating all pics in the ./ dir in a 4 row and 4 col, call as: 
python3 image-collages.py ./ 4 4

Output is stored in the cwd.
'''

input_dir = argv[1]
collage_size = [0,0]
collage_size[0] = int(argv[2])
collage_size[1] = int(argv[3])

def create_collage(input_dir, collage_size):
    if collage_size == [0,0]:
        print('Fatal error, specify collage dimensions.')
        quit()
    r, c = collage_size
    images_outers= []
    album = os.listdir(str(input_dir))
    c0 = 0

    for i in range(r):
        images = []

        for image_name in album[c0:c0+c]:
            print(image_name)
            image = cv2.imread(os.path.join(input_dir, image_name))
            image = cv2.resize(image, (680,680))
            images.append(image)

        image_outer = np.hstack(images)
        images_outers.append(image_outer)

        c0 += c

    return np.vstack(images_outers)


img = create_collage(input_dir, collage_size)
cv2.imwrite('collage.png', img)
