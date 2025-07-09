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
Accuracy Comparison between the [original PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and our optimized variant, indicating that our optimization only requires a low precision loss.
|Task|Original ACC|Optimzed ACC|
|:---:|:---:|:---:|
|Classification|90.4|90.5|

### Speed Comparison
Speed Comparison between the [original PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and our optimized variant.
|Layer|Sub-block Partitioning|Sampling|Neighbor Search|Feature Update|Aggregation|All|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Layer 1|-|11.86×|1.46×|1.75×|1.39×|2.37×|
|Layer 2|-|2.90×|1.37×|13.50×|0.90×|3.39×|

Note: Additional overhead has been introduced to adapt to the Pytorch implementation, resulting in a weakened acceleration effect.

## Acknowledgment
This project is partially based on [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by yanx27, licensed under the MIT License (Copyright (c) 2019 benny).




