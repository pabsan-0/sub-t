
# This bash script will setup a folder structure to work with YOLOv4 networks
# Pretrained weights are included in the main dir
# Find commands for training at the bottom of this script
#
#  ~/
#  └── YOLOv4/
#      ├── darknet/
#      │   └── ...
#      ├── yolo_backup/
#      │
#      ├── yolov4-tiny.conv.29
#      ├── yolov4.conv.137
#      ├── yolov4-csp.conv.142
#      ├── yolov4x-mish.conv.166
#      │
#      ├── yolov4-tiny-416-6.cfg
#      ├── yolov4-416-6.cfg
#      ├── yolov4-csp-512-6.cfg
#      ├── yolov4x-mish-640-6.cfg
#      │
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


# Setup folder structure
mkdir ~/YOLOv4/
mkdir ~/YOLOv4/PPU-6
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
cd ~/YOLOv4/
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.conv.166


### Download the modified cfg files ### ------------------------------------------
cd ~/YOLOv4/

# Google drive IDs for the files...
# yolo4csp    1Ichrr6Uu3TRvKuGW2ZqYumGzCu21XhtG
# yolo4tiny   1ZZyaNh_bfiEXVknNqtJ9UHlvEReeQUKb
# yolo4mish   1m4xxAacLuQ7NR7qStu0pfGONBHgzwLcx
# yolo4       1wUWi3q1DlpfmLQpxuBdHx1TveWUYPihA

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
train = ~/YOLOv4/PPU-6/train.txt
valid = ~/YOLOv4/PPU-6/valid.txt
names = ~/YOLOv4/PPU-6/obj.names
backup = ~/YOLOv4/yolo_backup
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

# Generate a python script that will generate the txts
cat > ~/YOLOv4/PPU-6/generate-txts.py << ENDOFFILE
import os
trainpath = '~/YOLOv4/PPU-6/train/'
validpath = '~/YOLOv4/PPU-6/test/'
with open('./train.txt', 'w') as file:
    a = os.listdir('./train')
    a = [i for i in a if i[-1]!='t']
    for i in a:
        file.write(trainpath + i +'\n')
with open('./valid.txt', 'w') as file:
    a = os.listdir('./test')
    a = [i for i in a if i[-1]!='t']
    for i in a:
        file.write(validpath + i +'\n')
ENDOFFILE

# generate the txt targets and remove the python script
touch ~/YOLOv4/PPU-6/train.txt
touch ~/YOLOv4/PPU-6/valid.txt
python3 ~/YOLOv4/PPU-6/generate-txts.py
rm ~/YOLOv4/PPU-6/generate-txts.py



### Lines for actually training each model ### ------------------------------------------
### Retrieve network output from yolo_backup/
### Delete the whole dir to clear space once finished, nothing is written outside of it
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
