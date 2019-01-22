# Benckmark and Model Zoo

## Environmnet

### Hardware


- 8 NVIDIA Tesla V100 GPUs

- Intel Xeon 4114 CPU @ 2.20GHz



### Software environment

- Python 3.6 / 3.7
- Mxnet 1.3
- CUDA 9.0.176
- CUDNN 7.0.4
- NCCL 2.1.15



## Common setting

- All baseline were trained using 8 GPUs with a batch size of 16 (2 images per GPU).
- All models were trained on 'coco_2017_train', and tested on the 'coco_2017_val'.



### Faster RCNN

| Backbone | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50     | 1x      |          |                     |                |        |          |
| R-101    | 1x      |          |                     |                |        |          |


### FPN

| Backbone | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50     | 1x      |          |                     |                |        |          |
| R-101    | 1x      |          |                     |                |        |          |



### Mask RCNN

| Backbone | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:--------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50     | 1x      |          |                     |                |        |         |          |
| R-101    | 1x      |          |                     |                |        |         |          |


### RetinaNet

| Backbone | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50     | 1x      |          |                     |                |        |          |
| R-101    | 1x      |          |                     |                |        |          |
