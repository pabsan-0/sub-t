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
for index, logfile in enumerate(logfiles):

    # define holders
    mAP_list = []
    AP50_list = []
    AP75_list = []

    # go over each line until certain keywords are found, then extract values
    with open(logfile, 'r') as file:
        for idx, line in enumerate(file.readlines()):
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
                mAP = line.split('= ')[-1]
                mAP_list.append(float(mAP))

            if 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]' in line:
                AP50 = line.split('= ')[-1]
                AP50_list.append(float(AP50))

            if 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]' in line:
                AP75 = line.split('= ')[-1]
                AP75_list.append(float(AP75))


    # generate plot
    plt.figure(figsize=(8.5, 5))
    plt.grid(linestyle=':')
    plt.plot(mAP_list)
    plt.plot(AP50_list)
    plt.plot(AP75_list)

    plt.xlim([0,350])
    plt.ylim([0,0.7])

    # plot assets
    plt.legend(['mAP', 'AP@50', 'AP@75'])
    plt.ylabel('Precision')
    plt.xlabel('Iteration')
    plt.show()

    #plt.saveimg('/home/pablo/Desktop/img-{index}.png')
