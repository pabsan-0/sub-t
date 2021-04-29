# Online detections with darknet
Darknet includes python bindings in the script *darknet/darknet.py*. With these bindings, the network can be initializaed beforehand and the images inferred very quickly. The images however need a conversion to C image format. 

The following script holds a demo of how this can be done for a .jpeg image in the darknet directory: (NEEDS CLEANUP)

```
# move to darknet and call python interpreter by specifying the CUDNN shortcut
cd ~/YOLOv4/darknet

cat << EOF > demo_pablo_pythonscript.py
import cv2
import numpy as np
import ctypes

def numpy2darknet_img(image): # v1
    image = (image / 255.).astype(np.float32)
    # image = image[..., ::-1]

    assert (image.shape[2] == 3)

    # convert rgb image(h, w, c) to bgr (c, h, w)
    # image = np.flip(image, 2)
    data = np.swapaxes(image, 2, 1)
    data = np.swapaxes(data, 1, 0)
    print("type: ", data.shape, data.dtype)

    data = np.ascontiguousarray(data)

    im = IMAGE(w=image.shape[1], h=image.shape[0], c=3, data=data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return im

# import the python wrapper
import darknet
from darknet import *


# load a nnet
config_file = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.cfg'
data_file = '/home/pablo/YOLOv4/darknet/cfg/coco.data'
weights = '/home/pablo/YOLOv4/darknet/cfg/yolov4-tiny.weights'

network, class_names, colors = load_network(config_file, data_file, weights, batch_size=1)


image = numpy2darknet_img(cv2.imread('demopic.jpeg'))
a = detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45)
print(a)

EOF

LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib python3 demo_pablo_pythonscript.py
```
