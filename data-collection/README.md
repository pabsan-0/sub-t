# Data collection:
Assets and resources used for the generation of different data towards object detection benchmarking.

## Data generation roadmap:
<img src="https://user-images.githubusercontent.com/63670587/123279853-8bfcfe80-d508-11eb-8eda-7818f8026efb.png" height="600">


# Datasets:  
## [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing):  
First baseline dataset with the DARPA artifacts of interest in both real and synthetic images, annotated in YOLO format. All of the images are square but size can vary, resize before using if the model doesn´t do it for you. This dataset is built from:
  - ~400 instances of the items backpack, drill, extinguisher and survivor borrowed from the [PST-RGB dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb).
  - ~400 instances of rope and helmet artifacts, in pictures taken by the author at the LTU facilities. 
  - A set of virtual pictures providing ~100 instances of each artifact produced with Unity [SynthDet](https://github.com/Unity-Technologies/SynthDet).
 

| **PPU-6 Split** | Images   | Backpack | Helmet | Drill | Extinguisher | Survivor | Rope |  
|:------          |:-------: |:-------: |:-----: |:-----:|:------------:|:--------:|:----:| 
| **Train**       | 912      | 320      | 336    | 306   | 317          |  295     | 309  |  
| **Test**        | 520      | 152      | 158    | 142   | 155          | 146      | 137  |  
| **Valid**       | 275      | 81       | 80     | 80    | 77           | 76       | 80   |  



## [PP-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing): 
Child of the PPU-6 dataset, in which synthetic training pictures have been removed. All of the images are square but size can vary, resize before using if the model doesn´t do it for you. **The download link leads to the PPU-6 dataset, remove synthetic pictures after download.**

| **PP-6 Split**   | Backpack | Helmet | Drill | Extinguisher | Survivor | Rope |  
|:------           |:-------: |:-----: |:-----:|:------------:|:--------:|:----:| 
| **Train**        | 223      | 237    | 198   | 208          |  212     | 209  |  
| **Test**         | 152      | 158    | 142   | 155          | 146      | 137  |  
| **Valid**        | 81       | 80     | 80    | 77           | 76       | 80   |  



## [Unity-6-1000 dataset](https://drive.google.com/file/d/1jViuZrzWHTOIWU8SYt8pWVQY3mgi9aYC/view?usp=sharing): 
1000 sythetic images generated with Synthdet for the artifacts Backpack, Helmet, Drill, Extinguisher, Survivor and Rope.
