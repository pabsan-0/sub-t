import pandas
import numpy as np

'''
This is a helper script meant for manually extracting a balanced dataset
from a collection of YOLO pictures+annotations.

Requires a prepared csv file!
'''
filename = 'balancedDatasetList_Shuffle.txt'
df = pandas.read_csv('PSTR900-details.csv')

onlydrill           = df[df.onlyDrill==1]
onlyextinguisher    = df[df.onlyExtinguisher==1]
onlysurvivor        = df[df.onlySurvivor==1]
manyitems           = df[df.manyItems==1]

how_many_drill        = 540 - 302
how_many_extinguisher = 540 - 460
how_many_survivor     = 540 - 399

a = list(np.random.choice(onlydrill.picname,         how_many_drill))
b = list(np.random.choice(onlyextinguisher.picname,  how_many_extinguisher))
c = list(np.random.choice(onlysurvivor.picname,      how_many_survivor))
d = list(manyitems.picname)

with open(filename, 'w') as file:
    for namelist in [a,b,c, d]:
        for name in namelist:
            file.write(name + '\n')
