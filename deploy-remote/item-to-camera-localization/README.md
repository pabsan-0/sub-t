# Item to camera localization

This folder covers the task of detecting where an item is with respect to the camera.

## In this directory:
- [locate_items_demo.py][]: This demo script is used to get real-world coordinates of items with respect to the camera coordinate frame in the horizontal plane given their width and pixel coordinates in video, specified by hand.
- [darknet-live-infer-plus-location.py][]: Infer livestream with darknet to detect specific items and use bbox width to get their real-world coordinates with respect to the camera.
- [darknet-live-infer-plus-location-v2.py][]: Same as previous, only plot the item on a 2D map relative to the camera.

[locate_items_demo.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/item-to-camera-localization/locate_items_demo.py
[darknet-live-infer-plus-location.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/item-to-camera-localization/darknet-live-infer-plus-location-v2.py
[darknet-live-infer-plus-location-v2.py]: https://github.com/solder-fumes-asthma/sub-t/blob/master/deploy-remote/item-to-camera-localization/darknet-live-infer-plus-location.py


## Math behind this approach
- Known item width (x2-x1), u1 and u2.
![WhatsApp Image 2021-04-30 at 15 03 35](https://user-images.githubusercontent.com/63670587/116699064-71555000-a9c5-11eb-9b90-7d03c3f7db02.jpeg)
