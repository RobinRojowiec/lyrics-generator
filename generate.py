"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: generate.py
Date: 13.04.2020

"""

import json

import torch
import torch.nn as nn
from torch.autograd import Variable

artist = Variable(torch.LongTensor([[1]]))
genre = Variable(torch.LongTensor([[0]]))
next_word = Variable(torch.LongTensor([[1]]))
length = 300
states = None
vocab_file = "data/id2char.vocab"

with open(vocab_file, "r", encoding="utf8") as json_file:
    char_vocab = json.load(json_file)

model = torch.load("data/lstm_model.pt", map_location=torch.device("cpu"))
model.eval()

# greedy decoding
tokens = []
temperature = 1.0

for _ in range(length):
    predicted_word, states = model.next_word(artist, genre, next_word, states)

    output_dist = nn.functional.softmax(predicted_word.view(-1).div(temperature), dim=0).data
    predicted_label = torch.multinomial(output_dist, 1)
    char_id = predicted_label.item()

    # stop at <end> symbol
    if char_id == 2:
        break

    tokens.append(char_vocab[str(char_id)])
    next_word = torch.LongTensor([[char_id]])

text = "".join(['<start>'] + tokens + ['<end>'])
print(text)
