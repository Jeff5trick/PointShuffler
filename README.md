# PointShuffler
This is the code of paper "PointShuffler: Accelerating Point Cloud Neural Networks on General-Purpose GPUs"
 
## Install
The latest codes are tested on CUDA 11.8  Python 3.8 and Pytorch 2.1

Compile the CUDA layers for PointShuffler:
```python setup.py install```

## Run
To run the accuracy and speed tests on Pytorch:
```python test_classification.py --log_dir pointnet2_cls_ssg```

## Result
### Accuracy Comparison
Accuracy comparison between the [original PointNet++](https://github.com/horizon-research/efficient-deep-learning-for-point-clouds) and our optimized variant, indicating that our optimization only requires a low precision loss.
|Task|Original|Ours|
|:---:|:---:|:---:|
|Classification|89.9|90.8|

### Execution latency
Execution latencies (ms) of our optimized variant.
|Layer|Sub-block Partitioning|Sampling|Feature Update|Neighbor Search|Aggregation|All|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Layer 1|0.090|0.068|0.415|0.126|0.064|0.764|
|Layer 2|0.088|0.065|0.418|0.132|0.063|0.768|

Note: Additional overhead has been introduced to adapt to the Pytorch implementation, resulting in a weakened acceleration effect.

## Acknowledgment
This project is partially based on [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by yanx27, licensed under the MIT License (Copyright (c) 2019 benny).




