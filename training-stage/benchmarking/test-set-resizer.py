from sys import argv
import os
import shutil
import cv2

'''Takes a YOLO annotated picture folder and makes a copy of it in the current path, 
resizing the pictures to a specified square resolution.

args:
    A source directory for extracting image originals
    Resolution 1
    Resolution 2... etc (optional)
'''

# import source test path from arg
test_path = argv[1]
if test_path[-1] != '/':
    test_path = test_path + '/'

# import target_resolutions from arg
target_resolutions = argv[2:]

# create dump folders for each resolution
for resolution in target_resolutions:
    try:
        os.mkdir('./resized-' + resolution)
    except FileExistsError:
        print('Catched FileExistsError! There is already a test dir with that resolution.')

# fetch yolo directory and copy items to the new ones, resizing only images
for file in os.listdir(test_path):
    for resolution in target_resolutions:
        if file[-1] == 't':
            shutil.copy(test_path + file, './resized-' + resolution + '/' + file)
        else:
            pic = cv2.imread(test_path + file)
            pic = cv2.resize(pic, (int(resolution), int(resolution)))
            cv2.imwrite('./resized-' + resolution + '/' + file[:-4] + '.jpg', pic)

for resolution in target_resolutions:
    with open(f'test-{resolution}.txt', 'w') as file:
        for picname in [picname for picname in os.listdir('./resized-' + resolution) if picname[-1]!='t']:
            file.write(os.getcwd() + '/resized-' + resolution + '/' + picname + '\n')
