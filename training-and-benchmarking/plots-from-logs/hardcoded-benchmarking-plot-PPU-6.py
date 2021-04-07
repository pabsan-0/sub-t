import matplotlib
import matplotlib.pyplot as plt
import numpy as np



labels = '''\
yolov4-tiny-416
yolov4-tiny-3l-416
yolov4-416
yolov4-csp-512
yolov4x-mish-640
efficientdet-d0-512
efficientdet-d1-640'''

str = '''\
.712	.946	.875
.516	.912	.532
.619	.931	.735
.544	.907	.602
.615	.922	.751
.304	.512	.350
.319	.551	.345'''

by_model = str.split('\n')
full_split = [str.split('\t') for str in by_model]
full_split = np.float_(full_split)

labels = labels.split('\n')
map =  [i[0] for i in full_split]
ap50 = [i[1] for i in full_split]
ap75 = [i[2] for i in full_split]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(13)

rects1 = ax.bar(x - width * 1.1,    map,  width, label='mAP')
rects2 = ax.bar(x,                  ap50, width, label='AP@50')
rects3 = ax.bar(x + width * 1.1,    ap75, width, label='AP@75')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision')
ax.set_title('Results on the PPU-6 dataset')
ax.set_yticks(np.arange(0,1,0.1))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=0)
ax.bar_label(rects2, padding=0)
ax.bar_label(rects3, padding=0)

fig.tight_layout()
plt.show()