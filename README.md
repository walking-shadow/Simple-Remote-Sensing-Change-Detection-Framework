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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

此项目是一个遥感变化检测的极简代码框架，希望能够帮助到刚进入变化检测领域的人快速实现自己的想法，代码主要包含以下特点：

- 使用albumentations库进行丰富的数据增强
- 使用wandb记录下每次实验的超参数、训练和验证指标、训练和验证结果图片，保证每次实验过后都可以详细的分析原因，并且也不用额外在其他地方记录超参数，调参的时候也更方便
- 使用torchmetrics库来快速并且正确地计算指标，避免了初期我单独计算每个batch的指标再求平均的错误
- 使用了warm-up、amp等基本的训练策略
- 某些变化检测数据集需要自己进行额外的数据处理，比如对一整幅遥感影像进行裁剪、把遥感图像随机分配到训练集、验证集和测试集等，写好了各种数据集处理的代码，基本足够各种情况下的使用了
- 用尽可能规范的方式写了详尽的注释，基本代码逻辑很容易看懂
- 代码量少，代码逻辑清晰，尽可能用简洁的代码进行了实现
- 变化检测模型是一个简单的孪生网络结构，损失函数就是简单的bce损失和dice损失的和
