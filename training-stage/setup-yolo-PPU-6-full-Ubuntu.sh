#!/bin/bash

# This bash script will setup a folder structure to work with YOLOv4 networks.
# You will NEED TO HARDCODE YOUR HOME USER, since DARKNET WONT EAT ~/* dirs
# Run this script through dos2unix or copy raw text for compatibility
# Find commands for training at the bottom of this script
#
#  ~/
#  └── YOLOv4/
#      ├── darknet/
#      │   └── ...
#      ├── yolo_backup/
#      ├── weights/
#      │   ├── yolov4-tiny.conv.29
#      │   ├── yolov4.conv.137
#      │   ├── yolov4-csp.conv.142
#      │   └── yolov4x-mish.conv.166
#      ├── cfg/
#      │   ├── yolov4-tiny-416-6.cfg
#      │   ├── yolov4-tiny-416-6-test.cfg *
#      │   ├── yolov4-416-6.cfg
#      │   ├── yolov4-416-6-test.cfg *
#      │   ├── yolov4-csp-512-6.cfg
#      │   ├── yolov4-csp-512-6-test.cfg *
#      │   ├── yolov4x-mish-640-6.cfg
#      │   └── yolov4x-mish-640-6-test.cfg *
#      └── PPU-6/
#          ├── train/
#          │   ├── picture001.jpg
#          │   ├── picture001.txt
#          │   └── ...
#          ├── valid/
#          │   ├── picture101.jpg
#          │   ├── picture101.txt
#          │   └── ...
#          ├── test/
#          │   ├── picture201.jpg
#          │   ├── picture201.txt
#          │   └── ...
#          ├── train.txt
#          ├── valid.txt
#          ├── obj.data
#          └── obj.names
#
#  * not automatically downloaded, read script for more info

# Setup folder structure
mkdir ~/YOLOv4/
mkdir ~/YOLOv4/PPU-6
mkdir ~/YOLOv4/cfg
mkdir ~/YOLOv4/weights
mkdir ~/YOLOv4/yolo_backup

### Cloning Darknet + YOLO weights ### ----------------------------------------
# clone darknet repo
cd ~/YOLOv4/
git clone https://github.com/AlexeyAB/darknet

# change makefile to have GPU and OPENCV enabled
cd ~/YOLOv4/darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

# verify CUDA
# /usr/local/cuda/bin/nvcc --version

# make darknet
make


# download pretrained weights, all from AlexeyAB's repo
cd ~/YOLOv4/weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.conv.166


### Download the modified cfg files ### ------------------------------------------
cd ~/YOLOv4/cfg

# Google drive IDs for the files...
# yolo4csp    1Ichrr6Uu3TRvKuGW2ZqYumGzCu21XhtG
# yolo4tiny   1ZZyaNh_bfiEXVknNqtJ9UHlvEReeQUKb
# yolo4mish   1m4xxAacLuQ7NR7qStu0pfGONBHgzwLcx
# yolo4       1wUWi3q1DlpfmLQpxuBdHx1TveWUYPihA
# Manually generate test cfg by setting batch size and subdivisions to 1

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ichrr6Uu3TRvKuGW2ZqYumGzCu21XhtG' -O yolov4-csp-512-6.cfg
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZZyaNh_bfiEXVknNqtJ9UHlvEReeQUKb' -O yolov4-tiny-416-6.cfg
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m4xxAacLuQ7NR7qStu0pfGONBHgzwLcx' -O yolov4x-mish-640-6.cfg
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wUWi3q1DlpfmLQpxuBdHx1TveWUYPihA' -O yolov4-416-6.cfg


### Obtain data assets, model-agnostic within YOLO ### ------------------------------------------
cd ~/YOLOv4/PPU-6

# download data, uncompress & clear the compressed file from disk
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep" -O PPU-6.rar && rm -rf /tmp/cookies.txt
unrar x -y PPU-6.rar
rm PPU-6.rar

# Generate obj.data
cat > ~/YOLOv4/PPU-6/obj.data << ENDOFFILE
classes = 6
train = /home/pablo/YOLOv4/PPU-6/train.txt
valid = /home/pablo/YOLOv4/PPU-6/valid.txt
names = /home/pablo/YOLOv4/PPU-6/obj.names
backup = /home/pablo/YOLOv4/yolo_backup
ENDOFFILE

# Generate obj.names
cat > ~/YOLOv4/PPU-6/obj.names << ENDOFFILE
backpack
helmet
drill
extinguisher
survivor
rope
ENDOFFILE

# Generate & run a python script that will generate the txts
cat > ~/YOLOv4/PPU-6/generate-txts.py << ENDOFFILE
import os
trainpath = '/home/pablo/YOLOv4/PPU-6/train/'
testpath  = '/home/pablo/YOLOv4/PPU-6/test/'
validpath = '/home/pablo/YOLOv4/PPU-6/valid/'
with open('./train.txt', 'w') as file:
    a = os.listdir('./train')
    a = [i for i in a if i[-1]!='t']
    for i in a:
        file.write(trainpath + i +'\n')
with open('./test.txt', 'w') as file:
    a = os.listdir('./test')
    a = [i for i in a if i[-1]!='t']
    for i in a:
        file.write(testpath + i +'\n')
with open('./valid.txt', 'w') as file:
    a = os.listdir('./valid')
    a = [i for i in a if i[-1]!='t']
    for i in a:
        file.write(validpath + i +'\n')
ENDOFFILE

# generate the txt targets and remove the python script
touch ~/YOLOv4/PPU-6/train.txt
touch ~/YOLOv4/PPU-6/test.txt
touch ~/YOLOv4/PPU-6/valid.txt
python3 ~/YOLOv4/PPU-6/generate-txts.py
rm ~/YOLOv4/PPU-6/generate-txts.py



### Lines for actually training each model ### ------------------------------------------
### Retrieve network output from yolo_backup/
### Delete the whole dir to clear space once finished, nothing is written outside of it
#
# LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4-tiny-416-6.cfg ~/YOLOv4/yolov4-tiny.conv.29 -dont_show -map
#
# Start training - change dir before!
# cd ~/YOLOv4/darknet/
#
# Train yolov4-tiny
# ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4-tiny-416-6.cfg ~/YOLOv4/yolov4-tiny.conv.29 -dont_show -map
#
# Train yolov4 vanilla
# ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4-416-6.cfg ~/YOLOv4/yolov4.conv.137 -dont_show -map
#
# Train yolov4-csp
# ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4-csp-512-6.cfg ~/YOLOv4/yolov4-csp.conv.142 -dont_show -map
#
# Train yolov4x-mish
# ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4x-mish-640-6.cfg ~/YOLOv4/yolov4x-mish.conv.166 -dont_show -map


# THE FOLLOWING ARE FOR MY LOCAL MACHINE, WHICH HAS ISSUES WITH CUDNN VERSIONS
# AND NEEDS TO DEFINE THE PATH VARIABLE BEFORE RUNNING THE COMMAND
#
# Train
# LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ ./darknet detector train ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/cfg/yolov4-tiny-416-6.cfg ~/YOLOv4/weights/yolov4-tiny.conv.29 -dont_show -map | tee ../custom_log.txt
#
# Test multiple files
# LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ ./darknet detector test cfg/obj.data cfg/yolov3.cfg yolov3.weights < images_files.txt
#
# Infer multilpe files with positional output to text file - USE THIS FOR MAP TOO
# LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ ./darknet detector test ~/YOLOv4/PPU-6/obj.data ~/YOLOv4/yolov4-tiny-416-6-test.cfg ~/YOLOv4/yolo_backup/yolov4-tiny-416-6_best.weights -dont_show -ext_output < ../PPU-6/test.txt > results_pablo.txt
