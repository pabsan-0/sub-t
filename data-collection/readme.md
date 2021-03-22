

# Datasets:  
## [PPU-6 dataset](https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing):  
First baseline dataset with the DARPA artifacts of interest in both real and synthetic images. It is built from:
  - ~400 instances of the items backpack, drill, extinguisher and survivor borrowed from the [PST-RGB dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb).
  - ~400 instances of rope and helmet artifacts, in pictures taken by the author at the LTU facilities. 
  - A set of virtual pictures providing ~100 instances of each artifact produced with [Unity perception](https://github.com/Unity-Technologies/com.unity.perception).
 

| PPU-6 Split | Backpack | Helmet | Drill | Extinguisher | Survivor | Rope |  
|:------      |:-------: |:-----: |:-----:|:------------:|:--------:|:----:| 
| Train       | 320      | 336    | 306   | 317          |  295     | 309  |  
| Test        | 152      | 158    | 142   | 155          | 146      | 137  |  
| Valid       | 81       | 80     | 80    | 77           | 76       | 80   |  