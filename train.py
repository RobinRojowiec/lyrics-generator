"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: train.py
Date: 13.04.2020

"""
import logging
import time

import numpy as np
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
    pin_memory=device_name == "cpu", collate_fn=lambda batch: combine(batch, 0, device, "char_id_length"))

# CE Loss (NLL + Softmax)
criterion = nn.CrossEntropyLoss().to(device)

# Init model
model = LSTMLyricsGenerator().to(device)
model.train()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def prepare_target(target_ids, lengths):
    lengths = lengths.squeeze(dim=1).squeeze(dim=1)
    out = torch.nn.utils.rnn.pack_padded_sequence(target_ids, lengths, batch_first=True)
    out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # unpack (back to padded)
    return out.reshape(out.size(0) * out.size(1))


epochs = 1000
best_loss = float('inf')
for epoch in range(1, epochs + 1):
    epoch_loss = .0
    start_time = time.time()
    total_batches = int(len(train_dataset) / batch_size)
    for train_index, batch_data in tqdm(enumerate(train_loader), total=total_batches):
        # zero gradients
        optimizer.zero_grad()
        model.zero_grad()

        # calculate loss
        logits, _ = model(**batch_data)
        target = prepare_target(batch_data["char_id_target_tensor"].squeeze(dim=1), batch_data["char_id_length"])
        loss = criterion(logits, target)
        loss.backward()

        epoch_loss += loss.detach().item()

        # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    if epoch_loss < best_loss:
        best_loss = epoch_loss

        train_dataset.save_vocabs()
        torch.save(model.state_dict(), "data/lstm_model.pt")

    print("Epoch {:1} Loss (Summed): {:2.4f}, Perplexity {:5.4f}".format(epoch, epoch_loss,
                                                                         np.exp(epoch_loss / total_batches)))
