from sys import argv
import matplotlib.pyplot as plt

'''
Plots the AP@0.50 for each training epoch of darknet models from the
logs of the platform. Allows multiple log input.

args:
    Log files that want to be plotted, as extracted from the darknet scripts.
'''

# reads name of as many log files as are provided as input
logfiles = []
for i in argv[1:]:
    logfiles.append(i)
    print(i)

# open each file and get a list with the APs over time during training
for logfile in logfiles:

    # define holders
    AP_list = []
    iter_list = []

    # go over each line until certain keywords are found, then extract values
    with open(logfile, 'r') as file:
        for idx, line in enumerate(file.readlines()):

            if 'next mAP calculation at' in line:
                current_iter = [int(i) for i in line.split() if i.isdigit()][0]

            if ', or' in line:
                AP = line.split(',')[0].split('=')[-1]

                # keep the values of interest
                iter_list.append(current_iter)
                AP_list.append(float(AP))

    # generate plot, matching epoch
    plt.plot(iter_list, AP_list)

# plot assets
plt.legend([logfile[:-4] for logfile in logfiles])
plt.ylabel('AP@0.50')
plt.xlabel('Iteration')
plt.show()
