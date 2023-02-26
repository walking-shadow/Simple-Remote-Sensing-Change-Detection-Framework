# deep-learning
all project related to deep learning

This project is a simplified implementation of remote sensing change detection based on pytorch, I hope it can help those who are beginners in change detection domain implementing their ideas quickly, without concerning other things. It has the following features:

- Using `albumentations` to implement abundant data augmentations.
- Using `wandb` to log hyper-parameters, metrics and images, so that we can analyse experiment and adjust hyper-parameter easily.
- Using `torchmetrics` to compute metrics quickly and properly.
- Using `warn up`, `amp`, and other basic training strategies.
- Many dataset process functions such as `crop image`, `random split image` are provided, as some change detection dataset needs to be processed with our own needs.
- Abundant annotations are provided in most of functions and classes.
- Code and its logic are very simple, easy to understand.
- Change detection model is a simple twin network, which served as an example, and loss function is just addition of BCELoss and DiceLoss.
