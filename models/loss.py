import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temp=0.5):
    """InfoNCE，z1/z2 是 [B, D] 的图级表征。"""
    z1, z2 = F.normalize(z1), F.normalize(z2)
    logits = torch.mm(z1, z2.t()) / temp          # [B, B]
    B = logits.size(0)
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

def task_loss(pred, y, task='classification'):
    if task == 'classification':
        return F.cross_entropy(pred, y)
    else:  # regression
        return F.mse_loss(pred.squeeze(), y)