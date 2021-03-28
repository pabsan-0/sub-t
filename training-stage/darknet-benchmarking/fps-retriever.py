from sys import argv
import numpy as np

'''
From a darknet results file, compute the average FPS of the predictions.

args:
    Output file
    Darknet results file 1
    Darknet results file 1 (optional)
    (...)
'''

output_file = argv[1]
with open(output_file, 'w'):
    pass

for darknet_results in argv[2:]:
    milliseconds = []
    with open(darknet_results, 'r') as file:
        for line in file.readlines():
            if 'Predicted in' in line:
                milliseconds.append(float(line.split('in ')[-1].split('milli')[0]))

    with open(output_file, 'a') as file:
        avg_ms = np.mean(milliseconds)
        file.write(f'{darknet_results}: {avg_ms} milli-seconds per image, or {1000/avg_ms} FPS\n')
