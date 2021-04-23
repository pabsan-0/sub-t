# Machine pose localization from camera input
This directory holds assets and test code for identifying the pose of a camera with respect to waypoints. 

- QR code:
  - Using pyzbar for localization and low-level opencv for image manipulation   
  - **DISCONTINUED in favor of ARUCO CODES**  
    - Hard to segmentate the code from the rest of the image  
    - Hard to read the code once segmentated  
- Aruco: 
  - Implemented inside opencv cv2.aruco for python3.
  - Higher level commands than for QR.
  - Codes are simpler and thus easier to identify and read.
  - **STATUS: relative poses between arucos can be found. Global camera position can be found. Orientation is too noisy.**
  - **TODO: Fix/look for alternatives on orientation sensors. Check how long is the reach. Check different lights.**



### External links
- [Python Open CV docs for camera calibration](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
- [Online Aruco marker generator](https://chev.me/arucogen/)
- [Some docs with cv2.aruco examples](https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/Projet+calibration-Paul.html)
- [GitHub repository with some example code](https://github.com/KyleJosling/QR-Pose-Position)
- [Rodrigues vectors and more basics explained](https://answers.opencv.org/question/215377/aruco-orientation-using-the-function-arucoestimateposesinglemarkers/)

![crop_aruco](https://user-images.githubusercontent.com/63670587/115121167-82f42c00-9fb1-11eb-8cf1-296ede9e99be.png)
![Screenshot from 2021-04-18 13-27-07](https://user-images.githubusercontent.com/63670587/115148801-8e9f2b80-a061-11eb-873e-d3434e3f2f78.png)
![WhatsApp Image 2021-04-23 at 15 55 41](https://user-images.githubusercontent.com/63670587/115903992-8df10580-a464-11eb-825d-b37ae7dd5438.jpeg)
