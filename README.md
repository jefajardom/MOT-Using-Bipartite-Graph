# Multi-Object-Tracking-Using-Bipartite-Graph

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cjl7xsJ3Kww3zPO1-d6us4i-Ho3ZsJ7g?usp=sharing)

The following has been addapted from the followin open source repositoris:
- https://github.com/dvl-tum/mot_neural_solver


### Video demos on MOT 16 challenge
<img src="output/mot_test_1.gif" width="400"/>  <img src="output/mot_test_8.gif" width="400"/> 

Left: MOT-16 set 1 | Right: MOT-16 set 8

## Abstract 

The following project is the development of a multi-object tracking (MOT) system, integrating the Faster R-CNN detector with a custom-build bipartite graph network. Utilizing the Hungarian algorithm for object matching, the system aims to address the complexities of tracking multiple objects in dynamic environments. My computational less demanding approach achieved a Multiple Object Tracking Accuracy (MOTA) of 58.8 and an ID F1 Score (IDF1) of 56.6. Future enhancements include the adoption of an offline tracking approach for better information utilization, the integration of a Kalman filter for improved prediction of bounding box positions, and the exploration of more advanced detection methods, such as YOLO, to further enhance performanc

## Setup 

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline

!pip install tqdm lap
!pip install https://github.com/timmeinhardt/py-motmetrics/archive/fix_pandas_deprecating_warnings.zip
!pip install git+https://github.com/cheind/py-motmetrics.git
```
## Results 

| Sequence    |  MOTA | IDF1 |  FP  |  FN  |
|-------------|-------|------|------|------|
|MOT16-2      | 49.6  | 48.6 |  390 | 8873 | 
|MOT16        |  77   |  70  |  266 | 1871 | 
|Overall      | 58.8  | 56.6 |  656 | 10744|

