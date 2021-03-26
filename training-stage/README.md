##  Training roadmap
![TRAIN-ROADMAP](https://user-images.githubusercontent.com/63670587/112643820-1029e200-8e45-11eb-8b6b-9b7c048f374d.png)


##  Training and inferring on the different platforms 
#### Working darknet models (YOLO family)
###### Use the file FILENAME to download everything you need for training darknet models on a local machine/google colab. This will set up a folder structure and download all required assets for setting up all the yolo networks, including the very PPU-6 dataset. Commands for training and inferring are included at the end of this file and should be manually run from a terminal window. Since batch inference with image output is not supported by default in AlexeyAB's branch of darknet, the procedure to follow for inferring and retrieving batches of labelled images is supported by a custom script.

- Training:
  - Start training for a default period of time (depends on model). 
  - Over time, new model weight files will be generated. A "best" weight file will be kept aside until a new one is generated with better AP on the valid set. 
  - Once training is finished, the optimal output weight file to use will be the "best" one. If validation AP hasnt't stalled (seen from plot), rerun training parting from these weights.

- Inference:
  - Generate a batch infer result text file with the command shown at the end of the file FILE.
  - Run the python script FILE to apply the labels contained in this text file to the pointed images and store them in a new directory.

- Benchmarking on COCO metrics:
  - Run the following command:
  - Then evaluate the output with the tools available at the (MS COCO repository)[https://github.com/cocodataset/cocoapi].
  
#### Working tensorflow models (EfficientDet)
###### Find the whole setup in the following [Colab Notebook](https://colab.research.google.com/drive/1mDyDFU5wtjKFR-EG05un8POHEaTs0W1B#scrollTo=uEG-D99zit7U).

- Training:
  - Start training for an infinite amount of iterations
  - Over time, new model checkpoint files will be generated. A "best" checkpoint file will be kept aside until a new one is generated with better AP on the valid set. 
  - Once the best checkpoint file hasn't been overwritten for a while and if its AP is good, cut training and keep this as the output model.

- Inference:
  - Convert the checkpoint directory to a .pb model.
  - Run the command shown in the colab notebook to extract the set of pictures.
  - Zip and import to drive to access the images.

- Benchmarking on COCO metrics:
  - EfficientDet already imports the MS COCO API for computing metrics, just use the command provided in the colab notebook on the checkpoint directory. 

##  Benchmarking on the [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing)

#### Full output + inferred test set
- YOLOv4-tiny: [yolov4-tiny-416-6](https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing)
- YOLOv4-tiny-3l: [yolov4-tiny-3l-416-6](https://drive.google.com/file/d/1qCwnTSipnOD12DV5JW_GnpsAzX_MVxtB/view?usp=sharing)
- YOLOv4: [yolov4-416-6](https://drive.google.com/file/d/1gs-wTb1AA3CxVfU7_mv0UDrvLsM0IHDT/view?usp=sharing)
- YOLOv4-csp: 
- YOLOv4x-mish:
- EfficientDet-D0: [effdet-d0-512-6](https://drive.google.com/file/d/1ngbk1b-gYV6nHC40hP6jXGsUmyMzChUM/view?usp=sharing)
- EfficientDet-D1:


#### Benchmarking summary, weights & sources
| Model           | Platform | Image size | Train time | mAP | AP@0.5 | AP@0.75 | FPS | Model weights  | 
|:-------------   |:------   | :---------:|:------:    |:---:|:---:|:---:|:---:|:--------   |
| yolov4-tiny     |Darknet   | 416x416    | 5h*        |     |||     |                | 
| yolov4-tiny-3l  |Darknet   | 416x416    | 5h*        |     |||     |                | 
| yolov4          |Darknet   |     |         |     |     |||                | 
| yolov4-csp      |Darknet   |     |         |     |     |||                | 
| yolov4x-mish    |Darknet   |     |         |     |     |||                | 
| efficientdet-d0 |Tensorflow|512x512|         |     |     |||                | 
| efficientdet-d1 |Tensorflow|     |         |     |     |||                | 

* NVIDIA GTX 1060 mobile




