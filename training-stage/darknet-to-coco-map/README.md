# Darknet to COCO for computing COCO metrics
Darknet output is not compatible with official pycocotools for obtaining mAP metrics official to the MS COCO challenge.
This folder includes assets to make the conversion from YOLO results to a json format that is compatible with pycocotools.

- darknet-2-coco-detections.py will transform the YOLO results into MS COCO formatted ones. Requires the output of darknet plus a COCO version of the ground-truths.
- valcoco.py holds a few commands for using the COCO API to compute the mAP table from the COCO json ground-truths and detections.
- PPU-6-test-coco.json holds the ground-truth labels of the PPU-6 test split, converted with the [tool in this repository](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter).
- The following assets were added to test and experiment with these tools:
  - yolov4-416-6-test-results-EXAMPLE.txt holds some darknet output to debug and test this tool.
  - The assets in the folder experimenting-id-0-1 are the GT and Dets for a single, fake image, to play around with numbers and see how the table changes.

To follow the whole thing, run these commands:
```
# Run darknet to get results  
detector test */obj.data *.cfg *.weights -dont_show -ext_output < */test.txt > results.txt

#Convert results to json  
python3 darknet_2_coco_detections.py PPU-6-test-coco.json yolov4-416-6-test-results.txt yolov4-416-6-coco.json

# Call valcoco.py to get the table  
python3 valcoco.py PPU-6-test-coco.json yolov4-416-6-coco.json
```
   
![Screenshot from 2021-03-27 15-57-01](https://user-images.githubusercontent.com/63670587/112724750-1be4d980-8f15-11eb-9888-bc9b1f29b189.png)

\* The fact that some -1 appears is because nAns have appeared when there are no items of that size (division by zero.)

## How did we arrive here
- COCO ground-truths template can be found in [the MS COCO webpage - data format](https://cocodataset.org/#format-data)
- COCO detection results can be seen in [the MS COCO webpage - results format](https://cocodataset.org/#format-results)
- Some actual result examples are provided in the [COCO repository](https://github.com/cocodataset/cocoapi/tree/master/results)
- [This tool](https://codebeautify.org/jsonviewer) proved very useful for previewing json files & working our way around them.
