## In this directory:
- 
-
-

## Working darknet models (YOLO family)
Use the file FILENAME to download everything you need for training darknet models on a local machine/google colab. This will set up a folder structure and download all required assets for setting up all the yolo networks, including the very PPU-6 dataset. Commands for training and inferring are included at the end of this file and should be manually run from a terminal window. Since batch inference with image output is not supported by default in AlexeyAB's branch of darknet, the procedure to follow for inferring and retrieving batches of labelled images is supported by a custom script.

- Training:
  - Start training for a default number of iterations. 
  - Over time, new model weight files will be generated. A "best" weight file will be kept aside until a new one is generated with better AP on the valid set. 
  - Once training is finished, the optimal output weight file to use will be the "best" one. If validation AP hasnt't stalled (seen from plot), rerun training parting from these weights.
- Inference:
  - Run the following command to export batch inference results into a text file:
  - Run the python script FILE to apply the labels contained in this text file to the pointed images and store them in a new directory.
- Benchmarking on COCO metrics:
  - Run the following command to export batch inference results into a text file:
  - Then evaluate the output with the tools available at the (MS COCO repository)[https://github.com/cocodataset/cocoapi].
  
