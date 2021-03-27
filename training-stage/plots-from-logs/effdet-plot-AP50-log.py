from sys import argv
import matplotlib.pyplot as plt

'''
Plots the AP@0.5 for each training epoch of efficient det models from the
logs of the platform. Allows multiple input

args:
    Log files that want to be plotted, as extracted from the effdet scripts.
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

    # go over each line until certain keywords are found, then extract values
    with open(logfile, 'r') as file:
        for idx, line in enumerate(file.readlines()):
            if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
                AP = line.split('= ')[-1]
                AP_list.append(float(AP))

    # generate plot
    plt.plot(AP_list)

# plot assets
plt.legend([logfile[:-4] for logfile in logfiles])
plt.ylabel('AP@0.50')
plt.xlabel('Iteration')
plt.show()
