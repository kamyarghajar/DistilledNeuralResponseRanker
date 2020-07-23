def v2t(x):
    itos = {v: k for k, v in tokenizer.vocab.items()}
    return ' '.join([itos[i.item()] if i.item() in itos else str(i.item()) for i in x if i != PAD])