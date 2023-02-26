# Simple-Remote-Sensing-Change-Detection-Framework
A simplified implementation of remote sensing change detection based on pytorch

This project is a simplified implementation of remote sensing change detection based on pytorch, I hope it can help those who are beginners in change detection domain implementing their ideas quickly, without concerning other things. It has the following features:

- Using `albumentations` to implement abundant data augmentations.
- Using `wandb` to log hyper-parameters, metrics and images, so that we can analyse experiment and adjust hyper-parameter easily.
- Using `torchmetrics` to compute metrics quickly and properly.
- Using `warn up`, `amp`, and other basic training strategies.
- Many dataset process functions such as `crop image`, `random split image` are provided, as some change detection dataset needs to be processed with our own needs.
- Abundant annotations are provided in most of functions and classes.
- Code and its logic are very simple, easy to understand.
- Change detection model is a simple twin network, which served as an example, and loss function is just addition of BCELoss and DiceLoss.


入门遥感深度学习变化检测研究的时候，看了很多变化检测论文的代码，发现代码开源的论文里面，一些很厉害的论文由于各种原因一般只会开放自己的模型代码，而其他比较水的论文代码基本逻辑很乱，也没什么注释，并且也基本不重视内容的记录。

现在做的研究告一段落，因此把自己的代码整理了一些，做出了一个深度学习变化检测的代码框架，之后有什么关于变化检测模型的想法能够在框架里面快速的实现，不用在各种其他的代码里面摸索来摸索去，代码主要包含以下特点：

- 使用albumentations库进行丰富的数据增强
- 使用wandb记录下每次实验的超参数、训练和验证指标、训练和验证结果图片，保证每次实验过后都可以详细的分析原因，并且也不用额外在其他地方记录超参数，调参的时候也更方便
- 使用torchmetrics库来快速并且正确地计算指标，避免了初期我单独计算每个batch的指标再求平均的错误
- 使用了warm-up、amp等基本的训练策略
- 某些变化检测数据集需要自己进行额外的数据处理，比如对一整幅遥感影像进行裁剪、把遥感图像随机分配到训练集、验证集和测试集等，写好了各种数据集处理的代码，基本足够各种情况下的使用了
- 用尽可能规范的方式写了详尽的注释，基本代码逻辑很容易看懂
- 代码量少，代码逻辑清晰，尽可能用简洁的代码进行了实现
- 变化检测模型是一个简单的孪生网络结构，损失函数就是简单的bce损失和dice损失的和
