# Deployment on server-based approach
Here you can find assets for implementing YOLO darknet networks that can infer over local networks...
- sending video data from a remote device
- receiving video and inferring on a host machine
- uploading the result to yet another local server

## Requires
- Raspberry Pi 3 model B (or similar)
- Camera
- Host computer
- LAN network

## Steps
- Connect Raspberry Pi to host via ethernet, configure wifi & enable camera via `pi@raspberry:$ sudo raspi-config`.
- Connect Raspberry Pi and host PC to wifi network and find the IP of both.
- Access to the Raspberry Pi via SSH with `$ ssh pi@192.168.0.102` and follow the steps in [this guide to stream camera to http local server](https://tutorials-raspberrypi.com/raspberry-pi-security-camera-livestream-setup/):
    - Preparatory steps
    ```
    $ sudo apt-get update
    $ sudo apt-get upgrade
    $ sudo apt-get install motion -y
        
    $ ls /dev/video*              # Check for video device (default will be video0)
    $ sudo modprobe bcm2835-v4l2  # To inmediately display camera
    ```
    - Check camera details witn `$ pi@raspberrypi:~ $ v4l2-ctl -V`
    - Open the motion config file with `$ sudo nano /etc/motion/motion.conf` and apply the following changes:
    ```
    # Start in daemon (background) mode and release terminal (default: off)
    daemon on
    ...
    # Restrict stream connections to localhost only (default: on)
    stream_localhost off
    ...
    # Target base directory for pictures and films
    # Recommended to use absolute path. (Default: current working directory)
    target_dir /home/pi/Monitor
    ...
    v4l2_palette 15     # Nummer aus der Tabelle davor entnehmen, 15 enstpricht YUYV
    ... 
    # Image width (pixels). Valid range: Camera dependent, default: 352 
    width 640 

    # Image height (pixels). Valid range: Camera dependent, default: 288 
    height 480 
    
    # Maximum number of frames to be captured per second. 
    # Valid range: 2-100. Default: 100 (almost no limit). 
    framerate 100 
    ```
    - In the same config file, optionally increase the limit streaming FPS for a masive speed boost.
    - In the same config file, optionally change the server port (default 8081).
    - Run `pi@raspberry:$ sudo nano /etc/default/motion` and change the content to `start_motion_daemon=yes`.
    - Run the following commands to set up the required directory:
    ```
    mkdir /home/pi/Monitor
    sudo chgrp motion /home/pi/Monitor
    chmod g+rwx /home/pi/Monitor
    ```
    - Start the streaming service with `$sudo service motion start`.
- Run the following command in the host computer to run darknet, infer incoming video and posting a new video in yet another local server.
    ```
    LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib ~/YOLOv4/darknet/darknet detector demo \
      ~/YOLOv4/darknet/cfg/coco.data \
      ~/YOLOv4/darknet/cfg/yolov4-tiny.cfg  \
      ~/YOLOv4/darknet/cfg/yolov4-tiny.weights \
      http://192.168.0.102:8081/video?dummy=param.mjpg -i 0 \
      -json_port 8070 -mjpeg_port 8090 -ext_output -dont_show```
- Once running, access to the following urls with a web browser:
  - 192.168.0.102:8081 (raspberrypi:8081): Raw camera livestream
  - 192.168.0.112:8070 (hostpc:8070): Json with inferring results
  - 192.168.0.112:8090 (hostpc:8090): Darknet output livestream
- Stop the raspberry streaming service with `pi@raspberry:$ sudo service motion stop` 
