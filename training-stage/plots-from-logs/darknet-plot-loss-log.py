from sys import argv
import matplotlib.pyplot as plt

'''
Plots the LOSS for each training epoch of darknet models from the
logs of the platform. Allows multiple log input.

args:
    Log files that want to be plotted, as extracted from the darknet scripts.
'''

# reads name of as many log files as are provided as input
logfiles = []
for i in argv[1:]:
    logfiles.append(i)
    print(i)

# open each file and get a list with the mAPs over time during training
for logfile in logfiles:

    # define holders
    loss_list = []

    # go over each line until certain keywords are found, then extract values
    with open(logfile, 'r') as file:
        for idx, line in enumerate(file.readlines()):

            if 'avg loss' in line:
                loss = line.replace('avg loss', ':').split(':')[1].split(',')
                loss = [float(i) for i in loss]

                # keep the values of interest
                loss_list.append((loss[0]+loss[1])/2)

    # generate plot
    plt.plot(loss_list)

# plot assets
plt.legend([logfile[:-4] for logfile in logfiles])
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.show()
