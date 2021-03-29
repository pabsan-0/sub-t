##  Training roadmap
![TRAIN-ROADMAP](https://user-images.githubusercontent.com/63670587/112643820-1029e200-8e45-11eb-8b6b-9b7c048f374d.png)

##  Model Benchmarking on the [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing)

#### Full output + inferred test set
- YOLOv4-tiny: [yolov4-tiny-416-6](https://drive.google.com/file/d/1kGqmUowvL5ePiV0n4fvkYvy-2fD0FYwi/view?usp=sharing)
- YOLOv4-tiny-3l: [yolov4-tiny-3l-416-6](https://drive.google.com/file/d/1qCwnTSipnOD12DV5JW_GnpsAzX_MVxtB/view?usp=sharing)
- YOLOv4: [yolov4-416-6](https://drive.google.com/file/d/1gs-wTb1AA3CxVfU7_mv0UDrvLsM0IHDT/view?usp=sharing)
- YOLOv4-csp: 
- YOLOv4x-mish:
- EfficientDet-D0: [effdet-d0-512-6](https://drive.google.com/file/d/1ngbk1b-gYV6nHC40hP6jXGsUmyMzChUM/view?usp=sharing)
- EfficientDet-D1:


#### Benchmarking summary, weights & sources
| Model           | Platform    | Image size | mAP |AP@0.5|AP@0.75| FPS |
|:-------------   |:------      | :---------:|:---:|:----:|:-----:|:---:| 
| yolov4-tiny     |Darknet      | 416x416    |.712 | .946 | .875  |197  | 
| yolov4-tiny-3l  |Darknet      | 416x416    |.516 | .912 | .532  |182  |  
| yolov4          |Darknet      | 416x416    |.619 | .931 | .735  |28   |
| yolov4-csp      |Darknet      | 512x512    |.544 | .907 | .602  |26   |
| yolov4x-mish    |Darknet      | 640x640    |.615 | .922 | .751  |9    |
| efficientdet-d0 |google-automl| 512x512    |.304 | .512 | .350  |51   |
| efficientdet-d1 |google-automl| 640x640    |.319 | .551 | .345  |23   |

\* FPS benchmarked on NVIDIA GTX 1060-mobile
  


