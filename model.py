"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: model.py
Date: 13.04.2020

"""
import torch
import torch.nn as nn


class LSTMLyricsGenerator(nn.Module):
    def __init__(self, n_genres, n_artists, vocab_size, lstm_input_size=150, lstm_hidden_size=200, embedding_size=50):
        super(LSTMLyricsGenerator, self).__init__()

        self.genre_embedding = nn.Embedding(n_genres + 1, embedding_size, padding_idx=0)
        self.artist_embedding = nn.Embedding(n_artists + 1, embedding_size, padding_idx=0)
        self.char_embedding = nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.predict = nn.Linear(lstm_hidden_size, vocab_size)

    def forward(self, artist_id=None, genre_id=None, char_id_tensor=None, char_id_length=None, states=None):
        char_embed = self.char_embedding(char_id_tensor.squeeze(1)[:, 1:])
        genre_embed = self.genre_embedding(genre_id.squeeze(1)[:, 1:])
        artist_embed = self.artist_embedding(artist_id.squeeze(1)[:, 1:])

        input_tensor = torch.cat((artist_embed, genre_embed, char_embed), dim=2)

        out, (h, c) = self.lstm(input_tensor, states)
        out_drop = self.dropout(out)
        predicted = self.predict(out_drop)

        return predicted

    def next_word(self, artist, genre, last_word, states):
        char_embed = self.char_embedding(last_word)
        genre_embed = self.genre_embedding(genre)
        artist_embed = self.artist_embedding(artist)

        input_tensor = torch.cat((artist_embed, genre_embed, char_embed), dim=2)

        out, (h, c) = self.lstm(input_tensor, states)
        out_drop = self.dropout(out)
        predicted = self.predict(out_drop)

        return predicted, states
