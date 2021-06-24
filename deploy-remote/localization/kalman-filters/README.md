# Kalman filtering for localization

This folder holds a series of testing scripts for implementing Kalman filters on the ArUco positioning data. These files are kept for record and reference but are an advancing work i.e. they are used to progressively reach a final solution.

## In this directory:
- [A1-kalman-offline.m][]: Kalman filter implemeted on MATLAB (imported from a previous work).
- [A2-kalman-offline.py][]: Conversion of the A1 model to python numpy - can only run batched data.
- [A3-kalman-online.py][]: Reimplementation of A3 so that measurements can be filtered in real-time.
- [B1-kalman-online-multi.py][]: Expands to model B to allow for variable number of aruco markers on sight .
- [B2-kalman-live-multi.py][]: Bugfixes on the previous version.
- [RedmiNote9Pro.npz][]: Calibration parameters of my camera. Use yours.
- [data1.txt][]: Some data extracted from a static ArUco observation to infer the sensor noise covariance matrix.

The final implementation of the Kalman filter adds an extra fix to the B2 model (to account for angle weighted average issues within the filter) and can be found in [../perception-layer-final][].

[A1-kalman-offline.m]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/A1-kalman-offline.m
[A2-kalman-offline.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/A2-kalman-offline.py
[A3-kalman-online.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/A3-kalman-online.py
[B1-kalman-online-multi.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/B1-kalman-online-multi.py
[B2-kalman-live-multi.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/B2-kalman-live-multi.py
[RedmiNote9Pro.npz]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/RedmiNote9Pro.npz
[data1.txt]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/localization/kalman-filters/data1.txt
[../perception-layer-final]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final

## Model briefing
#### MODEL A:
Model A assumes a single aruco marker. Noisy measurements can be obtained with the ARUCO library from the camera sensor.
- State vector: *X*, *Z* and *Yaw*: the angle of rotation measured in the XZ-plane (horizontal w.r. to the ARUCO marker 0); and *each of its derivatives*.
- Measurements; *X*, *Z*, and *Yaw*.
- Propagation model: input *u* is vector 0, state variations come from the model of the process noise, which is carefully eye-balled. Propagation matrix A related X(k+1) with X(k).
- Measurement model: C matrix to relate state and measurement vector. Measurement noise taken from covariances in experiment recorded to *'data1.txt'*, which is read to compute Rv on execution.

#### MODEL B:
Model B knows a set of markers with given poses (relative to one of them which serves as global reference) and it expects any number of marker to appear on-screen. The Kalman filter is modified online to adapt to variations in the sizes of Y, C, Rv and C, so that a "sensor fusion" (even though with the same sensor) is performed for an accurate estimation of the pose. Model B Kalman filter is equal to that of model A, with the exceptions of the scalable matrices:
- Measurement vector Y: which modifies it size by stacking different poses: (x0, z0, yaw0) -> (x0, z0, yaw0, x1, z1, yaw1).
- C matrix: By doing the Y broadcasting this way, C matrix can be scaled by simply vertically stacking the original C matrix for one measurement.
- Rv: under the hypothesis that the camera shows the same error for all aruco markers, the Rv is computed by diagonally stacking the original Rv matrix.
- Kalman filter gain L: Which is horizontally broadcasted by stacking L slices that belong to each aruco marker.


