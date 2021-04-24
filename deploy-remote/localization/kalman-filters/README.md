# Kalman filtering for localization

This folder holds a series of scripts for implementing Kalman filters on the ARUCO positioning data. These files are kept for record and reference but are an advancing work i.e. they are used to progressively reach a final application.

## MODEL A:
The model A assumes a single aruco marker. Noisy measurements can be obtained with the ARUCO library from the camera sensor.
- State vector: *X*, *Z* and *Yaw*: the angle of rotation measured in the XZ-plane (horizontal w.r. to the ARUCO marker 0); and *each of its derivatives*.
- Measurements; *X*, *Z*, and *Yaw*.
- Propagation model: input *u* is vector 0, state variations come from the model of the process noise, which is carefully eye-balled. Propagation matrix A related X(k+1) with X(k).
- Measurement model: C matrix to relate state and measurement vector. Measurement noise taken from covariances in experiment recorded to *'data1.txt'*.

### Depiction of the model A:
<img src="https://user-images.githubusercontent.com/63670587/115956596-efb17e00-a4fd-11eb-897a-353c3b12e874.jpeg" width="600">
