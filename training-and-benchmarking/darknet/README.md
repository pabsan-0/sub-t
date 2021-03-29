# Training and benchmarking darknet models (YOLO family)
## In this directory:
- [setup-yolo-PPU-6-full-Ubuntu.sh](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/setup-yolo-PPU-6-full-Ubuntu.sh): Run this script to clone darknet and set up a folder structure, downloading all required assets for setting up all the yolo networks, including the very PPU-6 dataset. Commands for training and inferring are included at the end of this file and should be manually run from a terminal window.  

- [darknet-benchmarking](https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/darknet/darknet-benchmarking): Holds a set of scritps to go through all the after-training tasks: inferring and benchmarking both mAP and FPS.

- [6-class-cfg](https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/darknet/6-class-cfg): This folder contains the .cfg files that define the darknet networks architecture. These have been modified from the ones provided at [AlexeyAB's repository](https://github.com/AlexeyAB/darknet) to fit a 6-class object detection task.

## Procedure for working darknet models
The scripts will lead the way, but find here an outline of what is happening:
- Training:
  - Start training for a default number of iterations. 
  - Over time, new model weight files will be generated. A "best" weight file will be kept aside until a new one is generated with better AP on the valid set. 
  - Once training is finished, the optimal output weight file to use will be the "best" one. If validation AP hasnt't stalled (seen from plot), manually rerun training parting from these weights.
- Inference:
  - Batch inference with image output is not supported by default in AlexeyAB's branch of darknet, the procedure to follow for inferring and retrieving batches of labelled images is done by exporting results to a text file then modifying the pictures based on what's recorded there. Run the following command to export batch inference results into a text file: 
  - Run the python script FILE to apply the labels contained in this text file to the pointed images and store them in a new directory.
- Benchmarking on COCO metrics:
  - Run the following command to export batch inference results into a text file:
  - Then evaluate the output with the tools available at the [MS COCO repository](https://github.com/cocodataset/cocoapi). This requires additional steps that can be found in the [darknet-benchmarking](https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/darknet/darknet-benchmarking) dir.
  
