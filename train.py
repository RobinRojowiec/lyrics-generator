"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: train.py
Date: 13.04.2020

"""
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
batch_size = 32
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device_name = "cpu"
device = torch.device(device_name)
print("Device: %s" % device)

# Load data and make batches
train_dataset = LyricsDataset("data/preprocessed_lyrics.csv",
                              limit=1000, device=device)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=0,
    pin_memory=device_name == "cpu", collate_fn=lambda batch: combine(batch, 0, device))

# CE Loss (NLL + Softmax)
criterion = nn.CrossEntropyLoss().to(device)

# Init model
model = LSTMLyricsGenerator(n_genres=10, vocab_size=40, n_artists=11483).to(device)
model.train()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
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

    print("Epoch loss {:2.4f}".format(epoch_loss))

torch.save(model, "data/lstm_model.pt")
