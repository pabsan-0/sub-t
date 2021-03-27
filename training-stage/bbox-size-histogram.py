import os
from sys import argv
import matplotlib.pyplot as plt

'''
Plot an histogram with the sizes of bounding boxes. Meant to later modify the
size thresholds for more detailed COCO mAP tables.

Args:
    folder_path: the folder in which there are pictures and yolo annotations
    baseline_dim (optional): to get real numbers instead of per-unit when computing areas.
'''

# import folderpath argument and check / so that a filename can be appended later
folder_path = argv[1]
if folder_path[-1] != '/':
    folder_path.append('/')

# import baseline_dim argument else default to 1
try:
    baseline_dim = int(argv[2])
except:
    baseline_dim = 1


# define a holder list
areas = []

# compute area for each bbox from YOLO annotations
annotations = [filename for filename in os.listdir(folder_path) if filename[-1]=='t']

# compute area for each bbox from YOLO annotations
for filename in annotations:
    with open(folder_path + filename, 'r') as file:
        for line in file.readlines():
            area = float(line.split()[3]) * float(line.split()[4])
            areas.append(area * (baseline_dim ** 2))

plt.title('Size of items in YOLO annotations of %s' % folder_path)
plt.hist(areas, bins=100)
plt.ylabel('Amount')
plt.xlabel('pixels*pixels')
plt.show()
