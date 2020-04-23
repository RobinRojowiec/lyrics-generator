"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: model.py
Date: 13.04.2020

"""
import torch
import torch.nn as nn


class LyricsGenerator(nn.Module):
    def __init__(self, device):
        super(LyricsGenerator, self).__init__()
        self.song_generator = TextGenerator(device, title_for_context=False, generator="LSTM", enforce_sorted=False)

    def forward(self, artist_id=None, genre_id=None, char_id_tensor=None, char_id_length=None, states=None,
                title_ids=None, title_id_length=None, **kwargs):
        """
        Generates title and lyrics simultaneously
        :param title_id_length:
        :param artist_id:
        :param genre_id:
        :param char_id_tensor:
        :param char_id_length:
        :param states:
        :param title_ids:
        :param kwargs:
        :return:
        """
        title_out, title_states = self.song_generator(artist_id=artist_id, genre_id=genre_id,
                                                      char_id_tensor=title_ids, char_id_length=title_id_length)

        lyrics_out, lyrics_states = self.song_generator(artist_id=artist_id, genre_id=genre_id,
                                                        char_id_tensor=char_id_tensor, char_id_length=char_id_length
                                                        , title_ids=title_ids, title_id_length=title_id_length)

        return title_out, title_states, lyrics_out, lyrics_states


class TextGenerator(nn.Module):
    def __init__(self, device, n_genres=3, vocab_size=10000, n_artists=30, lstm_input_size=300, lstm_hidden_size=200,
                 embedding_size=50, num_layers=1, generator="LSTM", title_for_context=True, enforce_sorted=True):
        super(TextGenerator, self).__init__()

        # add one because idx 0 is for padding
        self.pad_id = 0
        self.vocab_size = vocab_size + 1
        self.word_embedding_size = 100

        self.genre_embedding = nn.Embedding(n_genres + 1, embedding_size, padding_idx=self.pad_id)
        self.artist_embedding = nn.Embedding(n_artists + 1, embedding_size, padding_idx=self.pad_id)
        self.word_embedding = nn.Embedding(self.vocab_size, 2 * embedding_size, padding_idx=self.pad_id)

        if title_for_context:
            lstm_input_size += embedding_size
        self.title_for_context = title_for_context

        if generator == "LSTM":
            self.rnn = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers=num_layers)
        elif generator == "GRU":
            self.rnn = nn.GRU(lstm_input_size, lstm_hidden_size, num_layers=num_layers)
        elif generator == "RNN":
            self.rnn = nn.RNN(lstm_input_size, lstm_hidden_size, num_layers=num_layers)

        self.dropout = nn.Dropout(0.2)
        self.predict = nn.Linear(lstm_hidden_size, self.vocab_size)
        self.enforce_sorted = enforce_sorted
        self.device = device

    def forward(self, artist_id=None, genre_id=None, char_id_tensor=None, char_id_length=None, states=None,
                title_ids=None, **kwargs):
        word_embed = self.word_embedding(char_id_tensor.squeeze(1))
        genre_embed = self.genre_embedding(genre_id.squeeze(1))
        artist_embed = self.artist_embedding(artist_id.squeeze(1))

        if title_ids is not None and self.title_for_context:
            title_embed = self.word_embedding(title_ids.squeeze(1))
            title_context_max = torch.max(title_embed, dim=1)[0].unsqueeze(dim=1)
            title_context_over_seq = torch.cat([title_context_max for _ in range(word_embed.size(1))], dim=1)
        else:
            title_context_over_seq = torch.zeros(genre_embed.size(0), genre_embed.size(1), self.word_embedding_size,
                                                 device=self.device)
        input_tensor = torch.cat((artist_embed, genre_embed, title_context_over_seq, word_embed), dim=2)

        if len(char_id_length.shape) > 1:
            lengths = char_id_length.squeeze(dim=1).squeeze(dim=1)
        else:
            lengths = char_id_length
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True,
                                                         enforce_sorted=self.enforce_sorted)
        out, states = self.rnn(packed, states)
        out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # unpack (back to padded)

        logits = self.dropout(self.predict(out))
        logits_flatten = logits.view(-1, self.vocab_size)
        return logits_flatten, states
