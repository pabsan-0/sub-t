# Kalman filtering for localization

This folder holds a series of scripts for implementing Kalman filters on the ARUCO positioning data. These files are kept for record and reference but are an advancing work i.e. they are used to progressively reach a final application.

## MODEL A:
Model A assumes a single aruco marker. Noisy measurements can be obtained with the ARUCO library from the camera sensor.
- State vector: *X*, *Z* and *Yaw*: the angle of rotation measured in the XZ-plane (horizontal w.r. to the ARUCO marker 0); and *each of its derivatives*.
- Measurements; *X*, *Z*, and *Yaw*.
- Propagation model: input *u* is vector 0, state variations come from the model of the process noise, which is carefully eye-balled. Propagation matrix A related X(k+1) with X(k).
- Measurement model: C matrix to relate state and measurement vector. Measurement noise taken from covariances in experiment recorded to *'data1.txt'*, which is read to compute Rv on execution.

## MODEL B:
Model B knows a set of markers with given poses (relative to one of them which serves as global reference) and it expects any number of marker to appear on-screen. The Kalman filter is modified online to adapt to variations in the sizes of Y, C, Rv and C, so that a "sensor fusion" (even though with the same sensor) is performed for an accurate estimation of the pose. Model B Kalman filter is equal to that of model A, with the exceptions of the scalable matrices:
- Measurement vector Y: which modifies it size by stacking different poses: (x0, z0, yaw0) -> (x0, z0, yaw0, x1, z1, yaw1).
- C matrix: By doing the Y broadcasting this way, C matrix can be scaled by simply vertically stacking the original C matrix for one measurement.
- Rv: under the hypothesis that the camera shows the same error for all aruco markers, the Rv is computed by diagonally stacking the original Rv matrix.
- Kalman filter gain L: Which is horizontally broadcasted by stacking L slices that belong to each aruco marker.

#### Depiction of the model A:
<img src="https://user-images.githubusercontent.com/63670587/115956596-efb17e00-a4fd-11eb-897a-353c3b12e874.jpeg" width="600">

#### Depiction of the model B:
<img src="" width="600">
