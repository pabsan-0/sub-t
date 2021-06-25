# sub-t
Object detection in Sub-T environments towards item search.

This repository hosts the files of the project *Convolutional neural networks for object detection in subterranean environments*, which aims to explore the capabilities of different state-of-the-art object detectors in the task of detecting some of the DARPA Sub-t Challenge artifacts from image data, then expand and propose a complete perception layer for item search in a mapped environment with a single camera.

***Written contents on the official report outweigh those in this repository.***

## Project breakdown
- **Research**: Gather knowledge on the field of neural networks and object detection. Identify state of the art object detection models and search for implementations backed-up by original research papers that are available for use.  
- **Data gathering**: Produce different datasets to train a model able to identify a specific set of items from pictures.   
- **Training and benchmarking**: Train different neural network models on the gathered data and evaluate and compare their performance on the object detection task.  
- **Deployment**: Build a perception layer suitable for item search around an object detection approach.

![image](https://user-images.githubusercontent.com/63670587/123279374-13963d80-d508-11eb-84a6-1cfa3f67ae62.png)



## Main outcomes and resources
- **Model benchmarking**: 
  - Benchmarking results: 
    - See [project report][] or showcase below
    - Full COCO tables available at [result-files][]
  - Generated data: 
    - Real samples: [PPU-6 dataset][]
    - Synthetic samples: [Unity-6-1000 dataset][]
  - Trained models: [available here][]
 
[project report]: None
[result-files]: https://github.com/pabsan-0/sub-t/tree/master/training-and-benchmarking/0-result-files-dump
[PPU-6 dataset]: https://drive.google.com/file/d/1D-oBYlsD2c4dWnMyhtav1_mYnqfNK-ep/view?usp=sharing
[Unity-6-1000 dataset]: https://drive.google.com/file/d/1jViuZrzWHTOIWU8SYt8pWVQY3mgi9aYC/view?usp=sharing
[available here]: https://drive.google.com/drive/folders/1OLD1uxc3tgps7nPPNuuovNihPEKamQzQ?usp=sharing

- **Perception layer for item search**
  - Find source code and instructions under [./deploy-remote/perception-layer-final][]

[./deploy-remote/perception-layer-final]: https://github.com/solder-fumes-asthma/sub-t/tree/master/deploy-remote/perception-layer-final



## Result showcase

### Object detection benchmarking
<img src="https://user-images.githubusercontent.com/63670587/123283663-e3509e00-d50b-11eb-923c-f57b1891f02d.png" height="650">

### HouseLivingRoom experimental layout 
<img src="https://user-images.githubusercontent.com/63670587/123277294-450e0980-d506-11eb-9faa-0aa23f441682.png" height="400">

### Live item search
https://user-images.githubusercontent.com/63670587/123239343-8e972e00-d4df-11eb-8c71-16db5db55ad4.mp4

