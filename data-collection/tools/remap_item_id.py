import os
from sys import argv

'''
This script is used to remap YOLO ids to different numbers if it is neccessary.
Takes as input a directory where both text files and images should be, then
modifies each text file changing the id number.

ATTENTION: This script will OVERWRITE YOUR FILES WITHOUT BACKUP
'''

dir = argv[1]
textfiles = [i for i in os.listdir(dir) if i[-3:]=='txt']
os.chdir(dir)

print(f'Starting work on {dir}')

for txt in textfiles:
    with open(txt, 'r+') as file:
        content = file.readlines()

    for i, line in enumerate(content):
        words = line.split()
        temp = int(words[0])
        temp -= 1
        words[0] = str(temp)
        content[i] = ' '.join(words) + '\n'

    with open(txt, 'w') as file:
        content = file.writelines(content)
