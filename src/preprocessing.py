from config import TOKENIZER, MAX_LEN
# import pandas as pd

def process_data(tweet, selected_text, sentiment, tokenizer=TOKENIZER, max_len=MAX_LEN):
    tweet = " " + " ".join(str(tweet).split(" "))
    selected_text = " " + " ".join(str(selected_text).split(" "))

    len_selected = len(selected_text) - 1
    idx1 = idx2 = None
    for idx, letter in enumerate(selected_text):
        if (tweet[idx] == selected_text[1]) and (" " + tweet[idx: idx+len_selected] == selected_text):
            idx1 = idx
            idx2 = idx1 + len_selected - 1
            break
    
    if idx1!=None and idx2!=None:
        char_targets = [0] * len(tweet)
        for i in range(idx1, idx2+1):
            char_targets[i] = 1
    else:
        char_targets = [1] * len(tweet)

    tok_tweet = tokenizer.encode(tweet)
    ids = tok_tweet.ids
    mask = tok_tweet.attention_mask
    type_ids = tok_tweet.type_ids

    target_idx = []
    for i, (offset1, offset2) in enumerate(tok_tweet.offsets):
        if sum(char_targets[offset1: offset2])>0:
            target_idx.append(i)

    start_target = target_idx[0]
    end_target = target_idx[-1]


    sentiment_ids = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }

    ids = [0] + [sentiment_ids[sentiment]] + [2] + [2] + ids + [2]
    mask = [1] * len(ids)
    type_ids = [0] * len(ids)
    offsets = [(0, 0)] * 4 + tok_tweet.offsets
    start_target += 4
    end_target += 4

    padding_len = max_len - len(ids)
    if padding_len>0:
        ids = ids + [1] * padding_len
        mask = mask + [0] * padding_len
        type_ids = type_ids + [0] * padding_len
        offsets = offsets + [(0, 0)] * padding_len

    return {
        'ids': ids,
        'mask': mask,
        'token_type_ids': type_ids,
        'targets_start': start_target,
        'targets_end': end_target,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': offsets,
        'padding_len': padding_len
    }

# if __name__ == "__main__":
#     df = pd.read_csv('input/train.csv')
#     print(process_data(df.text.values[0], df.selected_text.values[0], df.sentiment.values[0]))



