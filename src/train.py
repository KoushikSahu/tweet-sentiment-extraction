from tqdm import tqdm
import torch
from loss import loss_fn

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for d in tqdm(data_loader):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
