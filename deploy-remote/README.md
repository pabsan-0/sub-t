# Deploy remote

This directory holds assets and source code for a perception layer for item search baed on a convolutional object detector. 
This perception layer is to be run on a remote host machine interacting with the agent over a WiFi network.

**Most of what you'll find here is experimental code, go straigth to [percepion-layer-final](https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final) for the perception layer itself.**

## In this directory:
- [camera-calibration][]: Assets for camera callibration to get the camera matrix and compensate for distortion.
- [darknet-inference][]: Experimental work towards achieving inline real-time darknet inference
- [item-to-camera-localization][]: Experimental work towards item localization with respect to a camera from images.
- [localization][]: Experimental work towards camera localization with fiducial markers.
- [percepion-layer-final][]: Final implementation of the proposed perception layer. 

[camera-calibration]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/camera-calibration
[darknet-inference]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/darknet-inference
[item-to-camera-localization]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/item-to-camera-localization
[localization]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/localization
[percepion-layer-final]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final


## External links

- [IP webcam app][]: for sending phone camera feed over WiFi
[IP webcam app]: https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en&gl=US
