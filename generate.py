"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: generate.py
Date: 13.04.2020

"""
import torch

from data import LabelVocab

artist = 1
genre = 1
next_word = 1
length = 300
states = None

char_vocab = LabelVocab.load("data/chars.vocab")

model = torch.load("data/model")
model.eval()

text = ""
for _ in range(length):
    next_word, states = model.next_word(artist, genre, next_word, states)
    char_id = torch.max(next_word, dim=1)[0]

    text += char_vocab.get_label(char_id)

print(text)
