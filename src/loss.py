import torch.nn as nn

def loss_fn(o1, o2, t1, t2):
    l1 = nn.CrossEntropyLoss()(o1, t1.long())
    l2 = nn.CrossEntropyLoss()(o2, t2.long())
    return l1 + l2