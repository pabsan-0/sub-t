from sys import argv
import matplotlib.pyplot as plt

'''
Plots the mAP@0.50:0.95 for each trainin epoch of efficient det models from the
logs of the platform. Allows multiple input

args:
    Log files that want to be plotted, as extracted from the effdet scripts.
'''

# reads name of as many log files as are provided as input
logfiles = []
for i in argv[1:]:
    logfiles.append(i)
    print(i)

# open each file and get a list with the mAPs over time during training
for logfile in logfiles:

    # define holders
    mAP_list = []

    # go over each line until certain keywords are found, then extract values
    with open(logfile, 'r') as file:
        for idx, line in enumerate(file.readlines()):
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
                mAP = line.split('= ')[-1]
                mAP_list.append(float(mAP))

    # generate plot
    plt.plot(mAP_list)

# plot assets
plt.legend([logfile[:-4] for logfile in logfiles])
plt.ylabel('AP@0.50:0.05:0.95')
plt.xlabel('Iteration')
plt.show()
