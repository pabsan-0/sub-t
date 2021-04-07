import matplotlib
import matplotlib.pyplot as plt
import numpy as np


## BASELINE RESOLUTION ---------------------------------------------------------

labels = '''\
yolov4-tiny-416
yolov4-tiny-3l-416
yolov4-416
yolov4-csp-512'''

str = '''\
.576	.943	.628
.657	.935	.798
.559	.915	.644
.619	.915	.763'''

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
fig.set_figwidth(9)

rects1 = ax.bar(x - width * 1.1,    map,  width, label='mAP')
rects2 = ax.bar(x,                  ap50, width, label='AP@50')
rects3 = ax.bar(x + width * 1.1,    ap75, width, label='AP@75')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision')
ax.set_title('Results on the PP-6 dataset - baseline test size')
ax.set_yticks(np.arange(0,1,0.1))
ax.set_xticks(x)
ax.set_xticklabels([i + '\n(test-res %sx%s)' %(i.split('-')[-1],i.split('-')[-1]) for i in labels])
ax.legend(loc='best')

ax.bar_label(rects1, padding=0)
ax.bar_label(rects2, padding=0)
ax.bar_label(rects3, padding=0)

ax.axis(xmin=-0.6,xmax=4)
fig.tight_layout()





## FIXED RESOLUTION ------------------------------------------------------------


labels = '''\
yolov4-tiny-416
yolov4-tiny-3l-416
yolov4-416
yolov4-csp-512'''

str = '''\
.574	.946	.628
.668	.937	.818
.568	.915	.648
.619	.911	.711'''

by_model = str.split('\n')
full_split = [str.split('\t') for str in by_model]
full_split = np.float_(full_split)

labels = labels.split('\n')
map =  [i[0] for i in full_split]
ap50 = [i[1] for i in full_split]
ap75 = [i[2] for i in full_split]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig2, ax2 = plt.subplots()
fig2.set_figheight(5)
fig2.set_figwidth(9)

rects1 = ax2.bar(x - width * 1.1,    map,  width, label='mAP')
rects2 = ax2.bar(x,                  ap50, width, label='AP@50')
rects3 = ax2.bar(x + width * 1.1,    ap75, width, label='AP@75')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Precision')
ax2.set_title('Results on the PP-6 dataset - 640x640 test size')
ax2.set_yticks(np.arange(0,1,0.1))
ax2.set_xticks(x)
ax2.set_xticklabels([i + '\n(test-res 640x640)' for i in labels])
ax2.legend(loc='best')

ax2.bar_label(rects1, padding=0)
ax2.bar_label(rects2, padding=0)
ax2.bar_label(rects3, padding=0)

ax2.axis(xmin=-0.6,xmax=4)
fig2.tight_layout()


plt.show()
