##  Training and inferring on the different platforms 
#### Working darknet models (YOLO family)
- Training: Use the file FILENAME to download everything you need for training darknet models on a local machine/google colab. This will set up a folder structure and download all required assets for setting up all the yolo networks, including the very PPU-6 dataset. Commands for training and inferring are included at the end of this file and should be manually run from a terminal window.  

- Inference: Since batch inference with image output is not supported by default in AlexeyAB's branch of darknet, the procedure to follow for inferring and retrieving batches of labelled images is the following:
  - Generate a batch infer result text file with the command shown at the end of the file FILE.
  - Run the python script FILE to apply the labels contained in this text file to the pointed images and store them in a new directory.

#### Working tensorflow models (EfficientDet)
##  Benchmarking on the [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing)

#### Full output + inferred test set
- YOLOv4-tiny: [yolov4-tiny-416-6](https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing)
- YOLOv4: 
- YOLOv4-csp: 
- YOLOv4x-mish:
- EfficientDet:


#### Benchmarking summary, weights & sources
| Model        | Platform | Image size | Train time | mAP | FPS | Model weights  | 
|:-------------|:------   | :---------:|:------:    |:---:|:---:|:--------   |
| yolov4-tiny  |Darknet   | 416x416    | 5h*         |     |     |                | 
| yolov4       |Darknet   |     |         |     |     |                | 
| yolov4-csp   |Darknet   |     |         |     |     |                | 
| yolov4x-mish |Darknet   |     |         |     |     |                | 
| efficientdet |          |     |         |     |     |                | 

* NVIDIA GTX 1060 mobile




