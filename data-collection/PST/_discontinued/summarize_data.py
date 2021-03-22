import os
from sys import argv

'''
This script will read YOLO annotations and create a csv file telling
the number of instances of each item as well as the picture they appear in.
This log file will be generated in a hardcoded dir.

Add the annotations dir as first argument when calling this script.

DISCONTINUED
'''

# define csv name and path
csv = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PSTRGB_3000_resized/log_balanced.csv'

# read input dir, move there, pack all txt annotations to list
dir = argv[1]
os.chdir(dir)
textfiles = [i for i in os.listdir() if i[-3:]=='txt']

# create csv headers + annotate the target dir
with open(csv, 'w') as file:
    file.write(f'Dataset located at {dir}\n')
    file.write(f'Names:,backpack,helmet,drill,extinguisher,survivor,rope\n')
    file.write(f'IDs:,0,1,2,3,4,5\n')
    file.write('Cases:,=sum(B5:B8000),=sum(C5:C8000),=sum(D5:D8000),=sum(E5:E8000),=sum(F5:F8000),=sum(G5:G8000)\n')

# Run through each annotation and record the number of occurrences in the logfile
with open(csv, 'a+') as log:
    for txt in textfiles:
        with open(txt, 'r') as file:
            content = file.readlines()
            ids = [line.split()[0] for line in content]

            occurrences = [0,0,0,0,0,0]
            for i in [0,1,2,3,4,5]:
                occurrences[i] = sum([int(id) == i for id in ids])

            some_string = ",".join([str(i) for i in occurrences])
            log.write(f'{txt[:-4]}.png,{some_string}\n')
