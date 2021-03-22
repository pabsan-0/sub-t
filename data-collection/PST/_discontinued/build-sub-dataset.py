import shutil
'''
Builds a subdataset from a bigger one by reading the names of the desired files
from a txt. Retrieves both pictures and annotations from a YOLO annotated
data folder.
'''

textfile_name = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PSTRGB_3000_resized/balancedDatasetList_Shuffle.txt'
retrieveDir = 'D:/Workspace_win/sub-t-master/preparing-data/realdata/PSTRGB_3000_resized/PSTR900-YOLO-680x680/'
dumpDir =  'D:/Workspace_win/sub-t-master/preparing-data/realdata/PSTRGB_3000_resized/PSTR900-YOLO-680x680_balanced/'

with open(textfile_name, 'r') as file:
    piclist = file.readlines()

for pic in piclist:
    name = pic[:-5]
    shutil.copyfile(f'{retrieveDir + name}.png', f'{dumpDir + name}.png')
    shutil.copyfile(f'{retrieveDir + name}.txt', f'{dumpDir + name}.txt')
