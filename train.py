"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: train.py
Date: 13.04.2020

"""
import json
import logging
import time

import torch.utils.data
from torch import nn
# get command args
from tqdm import tqdm

from data import LyricsDataset, combine
from model import LSTMLyricsGenerator

# setup logging
logging.basicConfig(level=logging.INFO)
batch_size = 64
device_name = "cuda" if torch.cuda.is_available() else "cpu"
# device_name = "cpu"
device = torch.device(device_name)
print("Device: %s" % device)

# Load data and make batches
train_dataset = LyricsDataset("data/preprocessed_lyrics.csv", device=device)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=0,
    pin_memory=device_name == "cpu", collate_fn=lambda batch: combine(batch, 0, device))

# CE Loss (NLL + Softmax)
criterion = nn.CrossEntropyLoss().to(device)

# Init model
model = LSTMLyricsGenerator(n_genres=1, vocab_size=100, n_artists=2).to(device)
model.train()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
best_loss = float('inf')
for epoch in range(1, epochs + 1):
    epoch_loss = .0
    start_time = time.time()
    for train_index, batch_data in tqdm(enumerate(train_loader), total=int(
            len(train_dataset) / batch_size)):
        # zero gradients
        optimizer.zero_grad()
        model.zero_grad()

        # calculate loss
        logits = model(**batch_data)
        logits = logits.reshape(logits.size(0) * logits.size(1), logits.size(2))
        target = batch_data["char_id_tensor"].squeeze(1)[:, :-1].reshape(-1)
        loss = criterion(logits, target)
        loss.backward()

        epoch_loss += loss.detach().item()

        # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    if epoch_loss < best_loss:
        best_loss = epoch_loss

        with open("data/id2char.vocab", "w+", encoding="utf8") as vocab_file:
            json.dump(train_dataset.tokenizer.ids2char, vocab_file, ensure_ascii=False)

        with open("data/char2id.vocab", "w+", encoding="utf8") as vocab_file:
            json.dump(train_dataset.tokenizer.chars2id, vocab_file, ensure_ascii=False)

        torch.save(model, "data/lstm_model.pt")

    print("Epoch {:2} loss {:2.4f}".format(epoch + 1, epoch_loss))
