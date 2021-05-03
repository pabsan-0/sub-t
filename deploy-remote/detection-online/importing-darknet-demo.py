
'''
This is a DEMO script that shows how you can import darknet from wherever in your
computer, exclusively from python. It is not a clean solution, but makes no
permanent changes and works.
'''


# VERY NASTY way of importing the darknet bindings but it works
import os
import sys

# move to darknet dir and add the library path to sys.env
os.chdir('/home/pablo/YOLOv4/darknet')
sys.path.insert(1, '/home/pablo/YOLOv4/darknet')

try:
    import darknet
except OSError:
    # deals with the LD library path thing for CUDNN, run once per terminal execution
    print(''' Catched CUDNN error! Run this bash command and try again:
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib
    ''')


'''
Via interpreter you can follow these:

$ LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib python3
Python 3.8.6 (default, Jan 27 2021, 15:42:20)
[GCC 10.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import os
>>> os.chdir('/home/pablo/YOLOv4/darknet')
>>> import darknet
>>>
'''
