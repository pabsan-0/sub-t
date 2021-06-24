cd ~/YOLOv4/darknet

LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib ~/YOLOv4/darknet/darknet detector demo \
    ~/YOLOv4/darknet/cfg/coco.data \
    ~/YOLOv4/darknet/cfg/yolov4-tiny.cfg  \
    ~/YOLOv4/darknet/cfg/yolov4-tiny.weights \
    http://192.168.0.102:8081/video?dummy=param.mjpg -i 0 \
    -json_port 8070 -mjpeg_port 8090 -ext_output -dont_show
 
