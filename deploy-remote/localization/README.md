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
  - **STATUS: axis can be drawn on printed codes.**
  - **TODO: get relative pose between 2 codes and check that dimensions match reality.**


![crop_aruco](https://user-images.githubusercontent.com/63670587/115121167-82f42c00-9fb1-11eb-8cf1-296ede9e99be.png)
