# Final implementation of object localizer
This directory contains source code plus assets for running the proposed perception layer for item search.
For this, make sure all libraries and requisites are set up, hardcode the image source inside `main-multiprocessing.py` and then run the script normally from a terminal window.

## In this directory
- [RedmiNote9Pro.npz][]: File containing camera calibration parameters. Use yours.
- [data1.txt][]: (deprecated) Experimental data for estimating the sensor noise matrix towards Kalman filters.
- [main-multiprocessing.py][]: Main script calling the rest of the program. Holds some config under main.
- [main.py][]: (deprecated) Old version of main-multiprocessing, running the same source but sequentially instead of in parallel. You can use this, though it is slower and is a few commits behind.
- [pabloCameraPoseAruco.py][]: Custom python module implementing the Camera pose detection from ArUcos.
- [pabloDarknetInference.py][]: Custom python module implementing darknet inline inference.
- [pabloItem2CameraPosition.py][]: Custom python module implementing width-based item localization from bboxes. 
- [pabloKalman.py][]: (deprecated) Custom python module implementing the camera pose Kalman filter, buggy.
- [pabloKalman_yawEuclidean.py][]: Custom python module implementing the camera pose Kalman filter, debugged.
- [pabloWorldMap.py][]: Custom python module implementing likelihood map drawing to localize items in the environment.

[RedmiNote9Pro.npz]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/RedmiNote9Pro.npz
[data1.txt]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/data1.txt
[main-multiprocessing.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/main-multiprocessing.py
[main.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/main.py
[pabloCameraPoseAruco.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloCameraPoseAruco.py
[pabloDarknetInference.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloDarknetInference.py
[pabloItem2CameraPosition.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloItem2CameraPosition.py
[pabloKalman.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloKalman.py
[pabloKalman_yawEuclidean.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloKalman_yawEuclidean.py
[pabloWorldMap.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/perception-layer-final/pabloWorldMap.py


## To actually use the perception layer

- Make sure the following is properly installed
  - Darknet (with GPU support)
  - The python modules listed at `requirements.txt`
- Calibrate your camera with the tools at [../camera-calibration](https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/camera-calibration)
- Clone the [percepion-layer-final](https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final) directory (this directory).
  - Drop your camera's calibration file in the cloned dir
  - Check the mains of the file `main-multiprocessing.py` and hardcode-adjust:
    - Feed IP source
    - Camera calibration file
    - Darknet installation path
  - Run `main-multiprocessing.py` to start live-finding objects


## Media

### Sequential perception layer overview

### Multiprocessing perception layer overview



