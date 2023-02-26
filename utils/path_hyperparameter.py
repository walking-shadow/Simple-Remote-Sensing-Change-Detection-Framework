# Default Configs for train, val and test

class Path_Hyperparameter:
    random_seed = 360

    # training hyper-parameter
    epochs: int = 300  # Number of epochs
    batch_size: int = 16  # Batch size
    inference_ratio = 8  # batch_size in val and test equal to batch_size*inference_ratio
    learning_rate: float = 2e-4  # Learning rate
    factor = 0.1  # learning rate decreasing factor
    patience = 12  # schedular patience
    warm_up_step = 350  # warm up step
    weight_decay: float = 1e-3  # AdamW optimizer weight decay
    amp: bool = True  # if use mixed precision or not
    load: str = None  # Load model and/or optimizer from a .pth file for testing or continuing training
    max_norm: float = 20  # gradient clip max norm

    # evaluate and test hyper-parameter
    evaluate_epoch: int = 250  # start evaluate after training for evaluate epochs
    test_epoch: int = 200  # start test after training for test epochs
    stage_epoch = [0, 0, 0, 0, 0]  # adjust learning rate after every stage epoch
    save_checkpoint: bool = False  # if save checkpoint of model or not
    save_interval: int = 10  # save checkpoint every interval epoch
    save_best_model: bool = False  # if save best model or not

    # log wandb hyper-parameter
    log_wandb_project: str = 'dpcd_last2'  # wandb project name

    # data transform hyper-parameter
    noise_p: float = 0.8  # probability of adding noise

    # model hyper-parameter
    dropout_p: float = 0.3  # probability of dropout
    patch_size: int = 256  # size of input image

    # return all public attribute of class
    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}

ph = Path_Hyperparameter()
