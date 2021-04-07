# Benchmarking darknet models (YOLO family)
## In this directory:
- [darknet-benchmarking-all-nets.sh](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/darknet-benchmarking-all-nets.sh): Shell script to be run from bash to benchmark various darknet models. This makes use of a specific folder structure, review the script to check if it will match yours & adapt if neccessary. It will generate a series of temporary files that then will later move to a dump folder, leaving the FPS and MS COCO mAP results in this directory in text files. **For the purpose of benchmarking, this is the only file to be run, and wuill call other scripts on this folder itself.**

- [assets](https://github.com/solder-fumes-asthma/sub-t/tree/master/training-and-benchmarking/darknet/darknet-benchmarking/assets): Contains json files with the test split of the PPU-6 dataset.

- [batch-annotate.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/batch-annotate.py): This python script will annotate a series of pictures with a set of bounding boxes that will read from a darknet output text file.

- [darknet_2_coco_detections.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/darknet_2_coco_detections.py): Creates a json file in MS COCO format for predictions made with darknet.

- [fps-retriever.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/fps-retriever.py): From a darknet results text file, compute the average FPS of the predictions.

- [test-set-resizer.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/test-set-resizer.py): Takes a YOLO annotated picture folder and makes a copy of it in the current path, 
resizing the pictures to a specified square resolution. Used to analyze the networks with test pictures of its baseline size.

- [valcoco.py](https://github.com/solder-fumes-asthma/sub-t/blob/master/training-and-benchmarking/darknet/darknet-benchmarking/valcoco.py): Contains a few commands to analyze the MS COCO annotations and results and output a normalized MS COCO mAP table.


## External links:
- [MS COCO dataset website](https://cocodataset.org/#home)
- [MS COCO repository](https://github.com/cocodataset/cocoapi)
