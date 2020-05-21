import tokenizers

MAX_LEN = 192
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 5
ROBERTA_PATH = "/home/koushik/Documents/Pretrained Models/roberta-base"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)
TRAINING_FILE = 'input/train.csv'
MODEL_PATH = 'model'