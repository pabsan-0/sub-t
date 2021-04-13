# Training and benchmarking object detection models
Content for the training and benchmarking of the CNN-based oject detectors yolov4-tiny, yolo4-tiny-3l, yolov4, yolov4-csp, yolov4x-mish, efficientdet-d0 & efficientdet-d1 on a 6-class detection task.

[initial-roadmap-planning-deprecated](https://user-images.githubusercontent.com/63670587/112643820-1029e200-8e45-11eb-8b6b-9b7c048f374d.png)

<img src="https://user-images.githubusercontent.com/63670587/112826240-02ff3400-908d-11eb-8a42-51dafbdc650d.png" height="700">


##  In this directory:
- [darknet][]: Holds assets and scripts for working with darknet models. 
- [efficientdet][]: Holds assets and scripts for working with efficient-det models.
- [plots-from-logs][]: Holds a series of scripts to plot various graphs to analyze training and results.
- [obj.names][]: Text file holding the names of the classes to be identified, in key order. Inherited from YOLO-format annotations.

[darknet]: https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/darknet
[efficientdet]: https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/efficientdet
[plots-from-logs]: https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/plots-from-logs
[obj.names]: https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/obj.names


## Roadmap:
- **Training all models on default PPU-6**:
  - Default out-of-the-box training on official models. 
  - Observations: 
    - Tiny models outperform regular sizes (overfitting suspicion)         ⟶ *retrain on only real data*
    - YOLOv4-tiny outperforms YOLOv4-tiny-3l                               ⟶ *deeper experiment with anchor sizes / layer attribution*
    - YOLOs blow efficientdets out of of the water on few-class detections ⟶ *discard efficientdets from further experiments*

- **Training some darknet models on PP-6**
  - Removed synthetic training data to check if overfitting was taking place in the bigger models.
  - Experimented with changing resolutions since this collection of models was smaller:
    - [Train size, Network size,  Test image size] = `[train_size, train_size, train_size]`
    - [Train size, Network size,  Test image size] = `[train_size, train_size, 640x640]`
    - [Train size, Network size,  Test image size] = `[train_size, 640x640, 640x640]`
  - Observations
    - Variying results, tiny models still predominant                       ⟶ *no overfitting they are best architecture-wise for this small 6-class task*
    - YOLOv4-tiny-3l outperforms YOLOv4-tiny                                ⟶ *deeper experiment with anchor sizes / layer attribution*
    
- **Adjusting anchor sizes on best performing models**. Anchors not only impact expected aspect ratio but overall size. All yolo layers have to carry same anchors value, the `masks=` index is what decides what anchors are linked to each yolo layer
  - Modified anchors on yolov4-tiny (2 yolo layers + 6 anchor pairs). 
    - Anchors division: [0 1 2],[3 4 5].
    - Filters before each yolo layer: `filters = (classes + 5) * masks_per_layer`
    - Detected a mistake in the original yolov4-tiny.cfg because mask=0 was not attributed to any layer.


# Results
\* FPS benchmarked on NVIDIA GTX 1060-mobile  
\*\* FPS benchmarking on small networks can vary up to 20 FPS depending on simultaneous computer processes

## Benchmarking on the [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing)

| Model           | Platform    | Train size | mAP |AP@0.5|AP@0.75| FPS | Files + demo |
|:-------------   |:------      | :---------:|:---:|:----:|:-----:|:---:|:------ |
| yolov4-tiny     |Darknet      | 416x416    |**.712** | **.946** | **.875**  |**197**  |[yolov4-tiny-416-6][]|
| yolov4-tiny-3l  |Darknet      | 416x416    |.516 | .912 | .532  |182  |[yolov4-tiny-3l-416-6][]|
| yolov4          |Darknet      | 416x416    |.619 | .931 | .735  |28   |[yolov4-416-6][]|
| yolov4-csp      |Darknet      | 512x512    |.544 | .907 | .602  |26   |[yolov4-csp-512-6][]|
| yolov4x-mish    |Darknet      | 640x640    |.615 | .922 | .751  |9    |[yolov4x-mish-640-6][]|
| efficientdet-d0 |google-automl| 512x512    |.304 | .512 | .350  |51   |[effdet-d0-512-6][]|
| efficientdet-d1 |google-automl| 640x640    |.319 | .551 | .345  |23   |[effdet-d1-640-6][]|

[yolov4-tiny-416-6]: https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing
[yolov4-tiny-3l-416-6]: https://drive.google.com/file/d/1qCwnTSipnOD12DV5JW_GnpsAzX_MVxtB/view?usp=sharing
[yolov4-416-6]: https://drive.google.com/file/d/1gs-wTb1AA3CxVfU7_mv0UDrvLsM0IHDT/view?usp=sharing
[yolov4-csp-512-6]: https://drive.google.com/file/d/1GzztGVBPQjT8sqj8udEfFVUCaF6gCEX-/view?usp=sharing
[yolov4x-mish-640-6]: https://drive.google.com/file/d/1F4Fv2ENhwJ_QtK_FDB84PO1oP7ZiImJ_/view?usp=sharing
[effdet-d0-512-6]: https://drive.google.com/file/d/1ngbk1b-gYV6nHC40hP6jXGsUmyMzChUM/view?usp=sharing
[effdet-d1-640-6]: https://drive.google.com/file/d/1OV69bZeyq9pfkXmlMrGG4KsfX4ZMMQpD/view?usp=sharing

<img src="https://user-images.githubusercontent.com/63670587/113900765-74965b00-97ce-11eb-9e17-be0ff010c8b4.png" height="300">


## Benchmarking on the PP-6 dataset ([PPU-6](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing) without SynthDet samples). Removed synthetic data from training set and experimented different network and test image sizes.

#### Test size equals train size
| Model           | Platform    | Train size |Test size| mAP      |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:-------:| :---:    |:---:|:----:|:-----:|
| yolov4-tiny     |Darknet      | 416x416    | 416x416 |   .576   | **.943** | .628  |**184** | 
| yolov4-tiny-3l  |Darknet      | 416x416    | 416x416 | **.657** | .935 | **.798**  |183  |  
| yolov4          |Darknet      | 416x416    | 416x416 |   .559   | .915 | .644  |28   |
| yolov4-csp      |Darknet      | 512x512    | 512x512 |   .619   | .915 | .763  |26   |  

<img src="https://user-images.githubusercontent.com/63670587/113900759-73652e00-97ce-11eb-978c-cb6c536b9172.png" height="300">

#### Fixed test size - canon network size
| Model           | Platform    | Train size |Network size | Test size| mAP |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:-------:    |:-------:| :---:|:---:|:----:|:-----:|
| yolov4-tiny     |Darknet      | 416x416    | 416x416     | 640x640 | .574 | **.946** | .628  |**194** | 
| yolov4-tiny-3l  |Darknet      | 416x416    | 416x416     | 640x640 | **.668** | .937 | **.818**  | 173  |  
| yolov4          |Darknet      | 416x416    | 416x416     | 640x640 |  .568 | .915 | .648  |28   |
| yolov4-csp      |Darknet      | 512x512    | 512x512     | 640x640 | .619 | .911 | .711  |26   |

<img src="https://user-images.githubusercontent.com/63670587/113900764-73fdc480-97ce-11eb-9629-cebc75e1ad7b.png" height="300">


#### Fixed test size - higher network size
| Model           | Platform    | Train size | Network size |Test size| mAP |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:-------:     |:-------:| :---:|:---:|:----:|:-----:|
| yolov4-tiny     |Darknet      | 416x416    | 640x640      |640x640  | .503 | .897 | .519 | 102  | 
| yolov4-tiny-3l  |Darknet      | 416x416    | 640x640      |640x640  | .592 | .896 | .722 | 97   |  
| yolov4          |Darknet      | 416x416    | 640x640      |640x640  | .499 | .874 | .513 | 13   |
| yolov4-csp      |Darknet      | 512x512    | 640x640      |640x640  | .601 | .914 | .718 | 16   |


## Benchmarking tiny models on the ([PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing) without SynthDet samples). Adjusted anchors to fit our training data instead of recycling the MS COCO default ones.

| Model           | Platform    | Network size | mAP     | AP@0.5| AP@0.75 | 
|:-------------   |:------      | :---------:|:-------:| :---: |:---: |
| yolov4-tiny     |Darknet      | 416x416    | **.692**    | **.942**  | **.829** | 
| yolov4-tiny-3l  |Darknet      | 416x416    | .658    | .929  | .817 | 
