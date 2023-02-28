# Simple-Remote-Sensing-Change-Detection-Framework

## Introduction

This project is a simplified implementation of remote sensing change detection based on pytorch, I hope it can help those who are beginners in change detection domain implementing their ideas quickly, without concerning other things. It has the following features:

- Using `albumentations` to implement abundant data augmentations.
- Using `wandb` to log hyper-parameters, metrics and images, so that we can analyse experiment and adjust hyper-parameter easily.
- Using `torchmetrics` to compute metrics quickly and properly.
- Using `warn up`, `amp`, and other basic training strategies.
- Many dataset process functions such as `crop image`, `random split image` are provided, as some change detection dataset needs to be processed with our own needs.
- Abundant annotations are provided in most of functions and classes.
- Code and its logic are very simple, easy to understand.
- Change detection model is a simple twin network, which served as an example, and loss function is just addition of BCELoss and DiceLoss.

## Install dependencies

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Install Pytorch 1.12 or later](https://pytorch.org/get-started/locally/)
3. Install dependencies

​	Return the following code in command line.

`	pip install -r requirements.txt`

## Data

Using any change detection dataset you want, but organize dataset path as follows. `dataset_name`  is name of change detection dataset, you can set whatever you want.

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

Below are some binary change detection dataset you may want.

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)

Paper: A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[LEVIR-CD+](http://rs.ia.ac.cn/cp/portal/dataDetail?name=LEVIR-CD%2B)

[GoogleMap](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

Paper: SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images

[SYSU-CD](https://hub.fastgit.org/liumency/SYSU-CD)

Paper: SYSU-CD: A new change detection dataset in "A Deeply-supervised Attention Metric-based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

Paper: CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS

[NJDS](https://drive.google.com/file/d/1cQRWORIgW-X2BaeRo1hvFj7vlQtwnmne/view?userstoinvite=infinitemabel.wq@gmail.com&ts=636c5f76&actionButton=1&pli=1)

Paper: Semantic feature-constrained multitask siamese network for building change detection in high-spatial-resolution remote sensing imagery

[S2Looking](https://github.com/S2Looking/Dataset)

Paper: S2Looking: A Satellite Side-Looking Dataset for Building Change Detection

## Start

For training, run the following code in command line.

`python train.py`

If you want to debug while training, run the following code in command line.

`python -m ipdb train.py`



For test and inference, run the following code in command line.

`python inference.py` 

---

## 简介

此项目是一个遥感变化检测的极简代码框架，希望能够帮助到刚进入变化检测领域的人快速实现自己的想法，代码主要包含以下特点：

- 使用albumentations库进行丰富的数据增强
- 使用wandb记录下每次实验的超参数、训练和验证指标、训练和验证结果图片，保证每次实验过后都可以详细的分析原因，并且也不用额外在其他地方记录超参数，调参的时候也更方便
- 使用torchmetrics库来快速并且正确地计算指标，避免了初期我单独计算每个batch的指标再求平均的错误
- 使用了warm-up、amp等基本的训练策略
- 某些变化检测数据集需要自己进行额外的数据处理，比如对一整幅遥感影像进行裁剪、把遥感图像随机分配到训练集、验证集和测试集等，写好了各种数据集处理的代码，基本足够各种情况下的使用了
- 用尽可能规范的方式写了详尽的注释，基本代码逻辑很容易看懂
- 代码量少，代码逻辑清晰，尽可能用简洁的代码进行了实现
- 变化检测模型是一个简单的孪生网络结构，损失函数就是简单的bce损失和dice损失的和

## 下载需要的库

1. [下载CUDA](https://developer.nvidia.com/cuda-downloads)
2. [下载1.12或者更新的pytorch](https://pytorch.org/get-started/locally/)
3. 下载其他需要的包

​	在命令行中运行下面的命令下载其他需要的包

`	pip install -r requirements.txt`

## 数据

你可以使用任何你想使用的变化检测数据集，但是文件组织方式需要按照下面的来。`dataset_name`是你设置的变化检测数据集的名字。

```python
dataset_name
├─train
│  ├─label
│  ├─t1
│  └─t2
├─val
│  ├─label
│  ├─t1
│  └─t2
└─test
    ├─label
    ├─t1
    └─t2
```

下面是一些你可能需要的二分类变化检测数据集。

[WHU Building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Paper: Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set

[DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)

Paper: A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images

[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Paper: A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection

[LEVIR-CD+](http://rs.ia.ac.cn/cp/portal/dataDetail?name=LEVIR-CD%2B)

[GoogleMap](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)

Paper: SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images

[SYSU-CD](https://hub.fastgit.org/liumency/SYSU-CD)

Paper: SYSU-CD: A new change detection dataset in "A Deeply-supervised Attention Metric-based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection"

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)

Paper: CHANGE DETECTION IN REMOTE SENSING IMAGES USING CONDITIONAL ADVERSARIAL NETWORKS

[NJDS](https://drive.google.com/file/d/1cQRWORIgW-X2BaeRo1hvFj7vlQtwnmne/view?userstoinvite=infinitemabel.wq@gmail.com&ts=636c5f76&actionButton=1&pli=1)

Paper: Semantic feature-constrained multitask siamese network for building change detection in high-spatial-resolution remote sensing imagery

[S2Looking](https://github.com/S2Looking/Dataset)

Paper: S2Looking: A Satellite Side-Looking Dataset for Building Change Detection

## 开始

在命令行中运行下面的代码来开始训练

`python train.py`

如果你想在训练的时候进行调试，在命令行中运行下面的命令

`python -m ipdb train.py`

在命令行中运行下面的代码来开始测试或者推理

`python inference.py` 

