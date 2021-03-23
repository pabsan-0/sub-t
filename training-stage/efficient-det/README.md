## Assets for training Efficient Det

#### In this folder
- yolo2tfrecord_fixes.py: This file was borrowed from [AlessionTonioni's repo](https://github.com/AlessioTonioni/tf-objdetector). It is used to convert the YOLO annotations of the available data into .tfrecord files that efficient-det's official implementation requires. A few small changes have been made on this file:  
  - Fixed issues with tensorflow versions related to library names (for example tf.python_io.** -> tf.io.**)
  - Added an extra line to manage image extensions of different length than 3 (for instance .jpeg). This is not automated and commented lines around 115 should be changed.
 
#### External
- Efficient-det training on PPU-6: [Colab Notebook](https://colab.research.google.com/drive/1mDyDFU5wtjKFR-EG05un8POHEaTs0W1B#scrollTo=uEG-D99zit7U).
- Official repository from where the code is taken: [Google's Automl](https://github.com/google/automl).
