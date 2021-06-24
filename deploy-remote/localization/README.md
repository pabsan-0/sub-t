# Localization from camera input
This directory holds assets and test code for identifying the pose of a camera with respect to a series of landmarks. 
The following were the most experimented with:

- QR codes:
  - Using pyzbar for localization and low-level opencv for image manipulation   
  - **DISCONTINUED in favor of ARUCO MARKERS**  
    - Hard to segmentate the code from the rest of the image  
    - Hard to read the code once segmentated  
- ArUco markers: 
  - Implemented inside opencv cv2.aruco for python3.
  - Higher level commands than for QR.
  - Codes are simpler and thus easier to identify and read.

## In this directory:
- [aruco-test][]: Holds code for testing ArUco pose estimation and its conversion to camera pose.
- [kalman-filter][]: Holds source code for Kalman filters to improve noisy localization.
- [arucodetect-keeper.py][]: Receive live feed and find aruco patterns on it, get camera/relative poses... etc.
- [arucodetect-tocsv.py][]: Receive live feed and store read camera pose over time in a csv file.

The most up-to-date versions of these were directly added to the final implementation at [../perception-layer-final][].

[aruco-test]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/localization/aruco-test
[kalman-filter]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/localization/kalman-filters
[arucodetect-keeper.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/arucodetect-keeper.py
[arucodetect-tocsv.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/arucodetect-tocsv.py
[../perception-layer-final]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final


## Pending improvements:
- Better camera calibration
- Including inertial sensors from phone: Up to now, the sample time was too low because the sensors followed an http request. With a tcp socket the sample time is enough to include the inertial sensors. Following code as example

    ```
    import time
    import socket

    host="192.168.0.110"
    port=5555

    s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((host, port))

    while 1:
        tic = time.time()
        message, address = s.recvfrom(5555)
        print(message, 'TIME: ', time.time() - tic)
    ```

## External links
- [Python Open CV docs for camera calibration](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
- [Online Aruco marker generator](https://chev.me/arucogen/)
- [Some docs with cv2.aruco examples](https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/Projet+calibration-Paul.html)
- [GitHub repository with some example code on QR pose estimation](https://github.com/KyleJosling/QR-Pose-Position)
- [Rodrigues vectors and more basics explained](https://answers.opencv.org/question/215377/aruco-orientation-using-the-function-arucoestimateposesinglemarkers/)


## Media
![crop_aruco](https://user-images.githubusercontent.com/63670587/115121167-82f42c00-9fb1-11eb-8cf1-296ede9e99be.png)
![Screenshot from 2021-04-18 13-27-07](https://user-images.githubusercontent.com/63670587/115148801-8e9f2b80-a061-11eb-873e-d3434e3f2f78.png)
![WhatsApp Image 2021-04-23 at 15 55 41](https://user-images.githubusercontent.com/63670587/115903992-8df10580-a464-11eb-825d-b37ae7dd5438.jpeg)
