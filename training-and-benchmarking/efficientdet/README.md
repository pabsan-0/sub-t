# Training and benchmarking google-automl models (Efficientdet family)

## In this directory:
- [efficientdet-local-setup.sh](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/efficient-det/efficientdet-local-setup.sh): Short bash script to clone google-automl and download the models efficientdet-d0 and efficientdet-d1.

- [efficientdet-local-testFPS.sh](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/efficient-det/efficientdet-local-testFPS.sh): Short bash script to run a fps test on local hardware for the models efficientdet-d0 and efficientdet-d1.

- [yolo2tfrecord_fixes.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/efficient-det/yolo2tfrecord_fixes.py): This file was borrowed from [AlessionTonioni's repo](https://github.com/AlessioTonioni/tf-objdetector). It is used to convert the YOLO annotations of the available data into .tfrecord files that efficientdet's official implementation requires. A few small changes have been made on this file:  
  - Fixed issues with tensorflow versions related to library names (for example tf.python_io.\* -> tf.io.\*) on google colab.
  - Added an extra line (~115) to manage image extensions of different length than 3 (for instance .jpeg).
 
## External links
- [Efficientdet-D0 training on PPU-6:Colab Notebook](https://colab.research.google.com/drive/1mDyDFU5wtjKFR-EG05un8POHEaTs0W1B?usp=sharing).
- [Efficientdet-D1 training on PPU-6 Colab Notebook](https://colab.research.google.com/drive/1yRbZ9QMOH8_0SPQwV7OmgBu8JEfS1_Yi?usp=sharing).
- [Google's Automl official repository](https://github.com/google/automl).


## Procedure for working google-automl models
The colab notebooks will lead the way, but find here an outline of what is happening:
- Training:
  - Start training for an infinite amount of iterations.
  - Over time, new model checkpoint files will be generated. A "best" checkpoint file will be kept aside until a new one is generated with better AP on the valid set. 
  - Once the best checkpoint file hasn't been overwritten for a while and if its AP is good, cut training and keep this as the output model.
- Inference:
  - Convert the checkpoint directory to a .pb model.
  - Run the command shown in the colab notebook to extract the set of pictures.
  - Zip and import to drive to access the images.
- Benchmarking on COCO metrics:
  - EfficientDet already imports the MS COCO API for computing metrics, use the commands provided in the colab notebook on the checkpoint directory.


