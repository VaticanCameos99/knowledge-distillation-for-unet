import torch

def dice_loss(output, gt, smooth = 1):
    output = output.clamp(min = 0, max = 1)
    intersection = torch.sum(gt*output)
    union = torch.sum(gt) + torch.sum(output)
    dice = 1 -((2*intersection+smooth) / (union + smooth))
    return dice

metrics = {
    'dice_loss': dice_loss,
    #IoU -> measures something similar to worst case performance
    #Dice -> measures something similar to average case performance
} 

