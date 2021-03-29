## Assets for training Efficient Det

#### In this folder
- yolo2tfrecord_fixes.py: This file was borrowed from [AlessionTonioni's repo](https://github.com/AlessioTonioni/tf-objdetector). It is used to convert the YOLO annotations of the available data into .tfrecord files that efficient-det's official implementation requires. A few small changes have been made on this file:  
  - Fixed issues with tensorflow versions related to library names (for example tf.python_io.** -> tf.io.**) on google colab.
  - Added an extra line to manage image extensions of different length than 3 (for instance .jpeg). This is not automated and commented lines around 115 should be changed.
 
#### External
- Efficient-det training on PPU-6: [Colab Notebook](https://colab.research.google.com/drive/1mDyDFU5wtjKFR-EG05un8POHEaTs0W1B#scrollTo=uEG-D99zit7U).
- Official repository from where the code is taken: [Google's Automl](https://github.com/google/automl).


## Working tensorflow models (EfficientDet)
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
