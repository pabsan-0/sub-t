# Training and benchmarking object detection models
##  Training roadmap
[overall-roadmap-planning](https://user-images.githubusercontent.com/63670587/112643820-1029e200-8e45-11eb-8b6b-9b7c048f374d.png)
![image](https://user-images.githubusercontent.com/63670587/112826240-02ff3400-908d-11eb-8a42-51dafbdc650d.png)

## Model benchmarking
### Benchmarking on the [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing)

| Model           | Platform    | Train size | mAP |AP@0.5|AP@0.75| FPS | Files + demo |
|:-------------   |:------      | :---------:|:---:|:----:|:-----:|:---:|:------ |
| yolov4-tiny     |Darknet      | 416x416    |**.712** | **.946** | **.875**  |**197**  |[yolov4-tiny-416-6](https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing)|
| yolov4-tiny-3l  |Darknet      | 416x416    |.516 | .912 | .532  |182  |[yolov4-tiny-3l-416-6](https://drive.google.com/file/d/1qCwnTSipnOD12DV5JW_GnpsAzX_MVxtB/view?usp=sharing)|
| yolov4          |Darknet      | 416x416    |.619 | .931 | .735  |28   |[yolov4-416-6](https://drive.google.com/file/d/1gs-wTb1AA3CxVfU7_mv0UDrvLsM0IHDT/view?usp=sharing)|
| yolov4-csp      |Darknet      | 512x512    |.544 | .907 | .602  |26   |[yolov4-csp-512-6](https://drive.google.com/file/d/1GzztGVBPQjT8sqj8udEfFVUCaF6gCEX-/view?usp=sharing)|
| yolov4x-mish    |Darknet      | 640x640    |.615 | .922 | .751  |9    |[yolov4x-mish-640-6](https://drive.google.com/file/d/1F4Fv2ENhwJ_QtK_FDB84PO1oP7ZiImJ_/view?usp=sharing)|
| efficientdet-d0 |google-automl| 512x512    |.304 | .512 | .350  |51   |[effdet-d0-512-6](https://drive.google.com/file/d/1ngbk1b-gYV6nHC40hP6jXGsUmyMzChUM/view?usp=sharing)|
| efficientdet-d1 |google-automl| 640x640    |.319 | .551 | .345  |23   |[effdet-d1-640-6](https://drive.google.com/file/d/1OV69bZeyq9pfkXmlMrGG4KsfX4ZMMQpD/view?usp=sharing)|

\* FPS benchmarked on NVIDIA GTX 1060-mobile

### Benchmarking on the PP-6 dataset ([PPU-6](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing) without SynthDet samples)

| Model           | Platform    | Train size |Test size| mAP |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:-------:| :---:|:---:|:----:|:-----:|:---:| 
| yolov4-tiny     |Darknet      | 416x416    | 416x416 | .576 | **.943** | .628  |**184** | 
| yolov4-tiny-3l  |Darknet      | 416x416    | 416x416 | **.657** | .935 | **.798**  |183  |  
| yolov4          |Darknet      | 416x416    | 416x416 | .559 | .915 | .644  |28   |
| yolov4-csp      |Darknet      | 512x512    | 416x416 | .619 | .915 | .763  |26   |

| Model           | Platform    | Train size |Test size| mAP |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:-------:| :---:|:---:|:----:|:-----:|:---:| 
| yolov4-tiny     |Darknet      | 416x416    | 640x640 | .574 | **.946** | .628  |**194** | 
| yolov4-tiny-3l  |Darknet      | 416x416    | 640x640 | **.668** | .937 | **.818**  | 173  |  
| yolov4          |Darknet      | 416x416    | 640x640 | .568 | .915 | .648  |28   |
| yolov4-csp      |Darknet      | 512x512    | 640x640 | .619 | .911 | .711  |26   |

\* FPS benchmarked on NVIDIA GTX 1060-mobile
\*\* FPS benchmarking on small networks can vary up to 20 FPS depending on simultaneous computer processes

#### Full output + inferred test set (ignore these)
- YOLOv4-tiny: [yolov4-tiny-416-6](https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing)
- YOLOv4-tiny-3l: [yolov4-tiny-3l-416-6](https://drive.google.com/file/d/1qCwnTSipnOD12DV5JW_GnpsAzX_MVxtB/view?usp=sharing)
- YOLOv4: [yolov4-416-6](https://drive.google.com/file/d/1gs-wTb1AA3CxVfU7_mv0UDrvLsM0IHDT/view?usp=sharing)
- YOLOv4-csp: [yolov4-csp-512-6](https://drive.google.com/file/d/1GzztGVBPQjT8sqj8udEfFVUCaF6gCEX-/view?usp=sharing)
- YOLOv4x-mish: [yolov4x-mish-640-6](https://drive.google.com/file/d/1F4Fv2ENhwJ_QtK_FDB84PO1oP7ZiImJ_/view?usp=sharing)
- EfficientDet-D0: [effdet-d0-512-6](https://drive.google.com/file/d/1ngbk1b-gYV6nHC40hP6jXGsUmyMzChUM/view?usp=sharing)
- EfficientDet-D1: [effdet-d1-512-6](https://drive.google.com/file/d/1OV69bZeyq9pfkXmlMrGG4KsfX4ZMMQpD/view?usp=sharing)
