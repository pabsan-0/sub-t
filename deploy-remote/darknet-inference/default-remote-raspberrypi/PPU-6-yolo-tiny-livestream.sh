LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib ~/YOLOv4/darknet/darknet detector demo \
    ~/YOLOv4/PPU-6/obj.data \
    ~/YOLOv4/cfg/yolov4-tiny-416-6-test.cfg \
    ~/YOLOv4/weights-trained/yolov4-tiny-416-6_best.weights \
    http://192.168.0.102:8081/video?dummy=param.mjpg -i 0
