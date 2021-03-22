import os

'''
This script will update each time you press enter the number of instances that
exist in a set of YOLO-annotated images+labels folders. This is meant to manually
construct, via moving files, a balanced dataset split.

Paths can be HARDCODED at the beggining of this script.
'''
names = ['train', 'test', 'valid']

parent_dir = 'D:\Workspace_win\sub-t-master\datasets-ready\ltu-ptr-6'.replace('\\', '/')
train_dir = parent_dir + '/train/'
test_dir  = parent_dir + '/test/'
valid_dir = parent_dir + '/valid/'

while 1:
    for idx, dir in enumerate([train_dir, test_dir, valid_dir]):
        os.chdir(dir)
        txts = [i for i in os.listdir(dir) if i[-3:]=='txt']
        occurrences = [0,0,0,0,0,0]
        if txts != None:
            for txt in txts:
                with open(txt, 'r') as file:
                    content = file.readlines()
                    ids = [line.split()[0] for line in content]

                    for i in [0,1,2,3,4,5]:
                        occurrences[i] += sum([int(id) == i for id in ids])

        print(names[idx])
        print(occurrences)
    print('----------------------------------------')
    dump = input('waiting for input...')
