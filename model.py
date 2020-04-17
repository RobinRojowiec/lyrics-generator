"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: model.py
Date: 13.04.2020

"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMLyricsGenerator(nn.Module):
    def __init__(self, n_genres=3, vocab_size=166 + 3, n_artists=30, lstm_input_size=300, lstm_hidden_size=600,
                 embedding_size=100,
                 num_layers=2):
        super(LSTMLyricsGenerator, self).__init__()

        # add one because idx 0 is for padding
        self.pad_id = 0
        self.vocab_size = vocab_size + 1

        self.genre_embedding = nn.Embedding(n_genres + 1, embedding_size, padding_idx=self.pad_id)
        self.artist_embedding = nn.Embedding(n_artists + 1, embedding_size, padding_idx=self.pad_id)
        self.char_embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_id)

        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Linear(lstm_hidden_size, self.vocab_size)

    def forward(self, artist_id=None, genre_id=None, char_id_tensor=None, char_id_length=None, states=None, **kwargs):
        char_embed = self.char_embedding(char_id_tensor.squeeze(1))
        genre_embed = self.genre_embedding(genre_id.squeeze(1))
        artist_embed = self.artist_embedding(artist_id.squeeze(1))

        input_tensor = Variable(torch.cat((artist_embed, genre_embed, char_embed), dim=2))

        if len(char_id_length.shape) > 1:
            lengths = char_id_length.squeeze(dim=1).squeeze(dim=1)
        else:
            lengths = char_id_length
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)
        out, states = self.lstm(packed, states)
        out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # unpack (back to padded)

        logits = self.dropout(self.predict(out))
        logits_flatten = logits.view(-1, self.vocab_size)
        return logits_flatten, states
