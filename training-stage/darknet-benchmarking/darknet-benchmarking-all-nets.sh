python3 test-set-resizer.py ~/YOLOv4/PPU-6/test/ 416 512 640

pushd ~/YOLOv4/darknet

# TINY
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-tiny-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-tiny-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-tiny-416.txt

# TINY 3L
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-tiny-3l-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-tiny-3l-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-tiny-3l-416.txt

# YOLOv4
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-vanilla-416-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-416-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-416.txt \
        > $OLDPWD/results-yolov4-416.txt

# YOLOv4 CSP
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4-csp-512-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4-csp-512-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-512.txt \
        > $OLDPWD/results-yolov4-csp-512.txt

# YOLOv4 MISH
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/cuda-10.2/targets/x86_64-linux/lib/ \
    ./darknet detector test \
        ~/YOLOv4/PPU-6/obj.data \
        ~/YOLOv4/cfg/yolov4x-mish-640-6-test.cfg \
        ~/YOLOv4/weights-trained/yolov4x-mish-640-6_best.weights \
        -dont_show -ext_output < $OLDPWD/test-640.txt \
        > $OLDPWD/results-yolov4x-mish-640.txt

popd



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



python3 batch-annotate.py \
    results-yolov4-tiny-416.txt \
    results-yolov4-tiny-3l-416.txt \
    results-yolov4-416.txt \
    results-yolov4-csp-512.txt \
    results-yolov4x-mish-640.txt


rm -r ./generated-files-dump
mkdir ./generated-files-dump
mv *.json ./generated-files-dump
mv *.txt ./generated-files-dump
mv resized-* ./generated-files-dump

mkdir ./infer_results
mv results* ./infer_results


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




python3 fps-retriever.py \
    FPS.txt \
    ./generated-files-dump/results-yolov4-tiny-416.txt \
    ./generated-files-dump/results-yolov4-tiny-3l-416.txt \
    ./generated-files-dump/results-yolov4-416.txt \
    ./generated-files-dump/results-yolov4-csp-512.txt \
    ./generated-files-dump/results-yolov4x-mish-640.txt


echo Finished benchmarking successfully!
