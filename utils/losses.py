import torch
import torch.nn as nn


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_bce_loss(nn.Module):

    def __init__(self):
        super(dice_bce_loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.binary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binary_dice(torch.sigmoid(scores.clone()), labels)
        bce_loss = self.bce_loss(scores.clone(), labels)
        return [diceloss, bce_loss]


def change_loss(scores, labels):
    if len(scores.shape) > 3:
        scores = scores.squeeze(1)
    if len(labels.shape) > 3:
        labels = labels.squeeze(1)
    """ for binary change detection task"""
    criterion_change = dice_bce_loss()

    # change loss
    loss_change = criterion_change(scores, labels)

    return loss_change
