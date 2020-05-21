import torch
import numpy as np
from jaccard import jaccard
import torch.nn as nn
from config import TOKENIZER

def eval_fn(data_loader, model, device, tokenizer=TOKENIZER):
    model.eval()
    all_ids = []
    start_idx = []
    end_idx = []
    orig_selected = []
    padding_len = []

    for d in data_loader:
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        selected_text = d['orig_selected']
        pad_len = d['padding_len']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        all_ids.append(ids.cpu().detach().numpy())
        start_idx.append(torch.sigmoid(o1).cpu().detach().numpy())
        end_idx.append(torch.sigmoid(o2).cpu().detach().numpy())
        orig_selected.extend(selected_text)
        padding_len.extend(pad_len)

    start_idx = np.vstack(start_idx)
    end_idx = np.vstack(end_idx)
    all_ids = np.vstack(all_ids)

    jaccards = []

    for i in range(0, len(start_idx)):
        start_logits = start_idx[i][3: -padding_len[i]]
        end_logits = end_idx[i][3: -padding_len[i]]
        this_id = all_ids[i][3: -padding_len[i]]

        idx1 = idx2 = None
        max_sum = 0
        for ii, s in enumerate(start_logits):
            for jj, e in enumerate(end_logits):
                if  s+e > max_sum:
                    max_sum = s+e
                    idx1 = ii
                    idx2 = jj

        this_id = this_id[idx1: idx2+1]
        predicted_text = tokenizer.decode(this_id)
        predicted_text = predicted_text.strip()
        sel_text = orig_selected[i].strip()

        jaccards.append(jaccard(predicted_text, sel_text))

    return np.mean(jaccards)