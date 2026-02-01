# PointShuffler
This is the code of the paper "PointShuffler: Accelerating Point Cloud Neural Networks on General-Purpose GPUs". We implemented our proposed method using CUDA to accelerate point cloud processing, and integrated these CUDA operators as PyTorch extensions. We also release our pre-trained weights at the path /log/classification/pointnet2_cls_ssg/checkpoints.
 
## Install
The latest codes are tested on CUDA 11.8  Python 3.8 and Pytorch 2.1

Compile the CUDA layers for PointShuffler:
```python setup.py install```

## Run
To run the accuracy and speed tests on Pytorch:
```python test_classification.py --log_dir pointnet2_cls_ssg```

## Result
### Accuracy Comparison
Accuracy comparison between the [original PointNet++](https://github.com/horizon-research/efficient-deep-learning-for-point-clouds) and the pre-trained model of our optimized variant, indicating that our optimization only requires a low precision loss.
|Task|Original|Ours|
|:---:|:---:|:---:|
|Classification|90.8|90.7|

### Execution latency
Execution latencies (ms) comparison between [a baseline with conventional CUDA optimization](https://github.com/facebookresearch/votenet) and our optimized variant.

Layer1:
|Layer|Sub-block Partitioning|Sampling|Feature Update|Neighbor Search|Aggregation|All|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Baseline|-|0.810|0.727|0.184|0.089|1.810|
|Ours|0.090|0.068|0.415|0.126|0.064|0.764|

Layer2:
|Layer|Sub-block Partitioning|Sampling|Feature Update|Neighbor Search|Aggregation|All|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Baseline|-|0.190|0.574|1.781|0.057|2.602|
|Ours|0.088|0.065|0.418|0.132|0.063|0.768|

Note: Additional overhead has been introduced to adapt to the Pytorch implementation, resulting in a weakened acceleration effect.




