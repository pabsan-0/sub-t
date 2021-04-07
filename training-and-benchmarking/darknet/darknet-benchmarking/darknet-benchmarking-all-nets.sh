# Full benchmarking of MAP and FPS for various darknet models

# Generate a resized copy of the test set for benchmarking the nets on their baseline resolution
test-set-resizer.py ~/YOLOv4/PPU-6/test/ 416 512 640

# Temporary cd to darknet dir
pushd ~/YOLOv4/darknet


# Darknet inference (results to text file) on TINY
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-tiny-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-tiny-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-tiny-416.txt

# Darknet inference (results to text file) on TINY 3L
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-tiny-3l-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-tiny-3l-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-tiny-3l-416.txt

# Darknet inference (results to text file) on YOLOv4
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-vanilla-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-416.txt

# Darknet inference (results to text file) on YOLOv4 CSP
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-csp-512-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-csp-512-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-512.txt \
        > $OLDPWD/results-yolov4-csp-512.txt

# Darknet inference (results to text file) on YOLOv4 MISH
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4x-mish-640-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4x-mish-640-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-640.txt \
        > $OLDPWD/results-yolov4x-mish-640.txt


# Pop back to previous dir (the one holding this script)
popd


# Convert inference results from darknet to MS COCO
# PPU-6-size.json is loaded only to provide the paths to the images
python3 darknet_2_coco_detections.py \
    ./assets/PPU-6-416.json \
    results-yolov4-tiny-416.txt \
    results-yolov4-tiny-416.json

python3 darknet_2_coco_detections.py \
    ./assets/PPU-6-416.json \
    results-yolov4-tiny-3l-416.txt \
    results-yolov4-tiny-3l-416.json

python3 darknet_2_coco_detections.py \
    ./assets/PPU-6-416.json \
    results-yolov4-416.txt \
    results-yolov4-416.json

python3 darknet_2_coco_detections.py \
    ./assets/PPU-6-512.json \
    results-yolov4-csp-512.txt \
    results-yolov4-csp-512.json

python3 darknet_2_coco_detections.py \
    ./assets/PPU-6-640.json \
    results-yolov4x-mish-640.txt \
    results-yolov4x-mish-640.json


# Convert inference results from darknet to MS COCO
python3 batch-annotate.py \
    results-yolov4-tiny-416.txt \
    results-yolov4-tiny-3l-416.txt \
    results-yolov4-416.txt \
    results-yolov4-csp-512.txt \
    results-yolov4x-mish-640.txt


# Dump new files into dump dir
rm -r ./generated-files-dump
mkdir ./generated-files-dump
mv *.json ./generated-files-dump
mv *.txt ./generated-files-dump
mv resized-* ./generated-files-dump
mkdir ./infer_results
mv results* ./infer_results


# Remove MAP-table if exists so it is overwritten & generate MAP table text file
rm MAP-table.txt

echo MAP TABLE FOR YOLOv4-TINY >> MAP-table.txt
python3 valcoco.py \
    ./assets/PPU-6-416.json \
    ./generated-files-dump/results-yolov4-tiny-416.json \
    >> MAP-table.txt

echo MAP TABLE FOR YOLOv4-TINY-3L >> MAP-table.txt
python3 valcoco.py \
    ./assets/PPU-6-416.json \
    ./generated-files-dump/results-yolov4-tiny-3l-416.json \
    >> MAP-table.txt

echo MAP TABLE FOR YOLOv4 >> MAP-table.txt
python3 valcoco.py \
    ./assets/PPU-6-416.json \
    ./generated-files-dump/results-yolov4-416.json \
    >> MAP-table.txt

echo MAP TABLE FOR YOLOv4-CSP >> MAP-table.txt
python3 valcoco.py \
    ./assets/PPU-6-512.json \
    ./generated-files-dump/results-yolov4-csp-512.json \
    >> MAP-table.txt

echo MAP TABLE FOR YOLOv4x-MISH >> MAP-table.txt
python3 valcoco.py \
    ./assets/PPU-6-640.json \
    ./generated-files-dump/results-yolov4x-mish-640.json \
    >> MAP-table.txt


# Get the average FPS from the inference logs
python3 fps-retriever.py \
    FPS.txt \
    ./generated-files-dump/results-yolov4-tiny-416.txt \
    ./generated-files-dump/results-yolov4-tiny-3l-416.txt \
    ./generated-files-dump/results-yolov4-416.txt \
    ./generated-files-dump/results-yolov4-csp-512.txt \
    ./generated-files-dump/results-yolov4x-mish-640.txt


# Finish message alert
echo Finished benchmarking successfully!
