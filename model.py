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
    def __init__(self, device, vocab_size=15000, data_embedding_size=20, embedding_size=100,
                 dropout=0.1, n_genres=3, n_artists=30, hidden_size=200, num_layers=1, generator="GRU"):
        super(LyricsGenerator, self).__init__()

        # add one because idx 0 is for padding
        self.pad_id = 0
        self.vocab_size = vocab_size + 1
        self.word_embedding_size = 100
        self.keyword_size = 1000
        self.dropout = nn.Dropout(dropout)
        self.batch_norm_title = nn.BatchNorm1d(hidden_size)
        self.batch_norm_text = nn.BatchNorm1d(hidden_size)

        # word encoding
        self.word_embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=self.pad_id)
        self.keyword_embedding = nn.Embedding(int(self.vocab_size / 2), embedding_size, padding_idx=self.pad_id)

        text_input_size = embedding_size * 3 + 2 * data_embedding_size

        # lyrics text
        if generator == "LSTM":
            self.lyrics_rnn = nn.LSTM(text_input_size, hidden_size, num_layers=num_layers)
        elif generator == "GRU":
            self.lyrics_rnn = nn.GRU(text_input_size, hidden_size, num_layers=num_layers)
        elif generator == "RNN":
            self.lyrics_rnn = nn.RNN(text_input_size, hidden_size, num_layers=num_layers)
        self.predict_text = nn.Linear(hidden_size, self.vocab_size)

        # lyrics title
        self.title_rnn = nn.GRU(text_input_size - embedding_size, hidden_size, num_layers=num_layers)
        self.title_predictor = nn.Linear(hidden_size, self.vocab_size)

        # data encoding
        self.genre_embedding = nn.Embedding(n_genres + 1, data_embedding_size, padding_idx=self.pad_id)
        self.artist_embedding = nn.Embedding(n_artists + 1, data_embedding_size, padding_idx=self.pad_id)

        self.device = device

    def forward(self, artist_id=None, genre_id=None, keyword_id=None, char_id_tensor=None, char_id_length=None,
                title_ids=None, title_id_length=None, output="all", **kwargs):
        """
        Generates title and lyrics simultaneously
        :param keyword_id:
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
        # get states if provided
        text_states, title_states = kwargs.get("text_states", None), kwargs.get("title_states", None)

        # get embedding for data and word
        keyword_embed = self.keyword_embedding(keyword_id.squeeze(1))
        word_embed = self.word_embedding(char_id_tensor.squeeze(1))
        genre_embed = self.genre_embedding(genre_id.squeeze(1))
        artist_embed = self.artist_embedding(artist_id.squeeze(1))

        if output in ["title", "all"]:
            title_input_tensor = torch.cat((artist_embed, genre_embed, keyword_embed, word_embed), dim=2)
            if len(title_id_length.shape) > 1:
                title_id_length = title_id_length.squeeze(dim=1).squeeze(dim=1)
            else:
                title_id_length = title_id_length
            packed = torch.nn.utils.rnn.pack_padded_sequence(title_input_tensor, title_id_length, batch_first=True,
                                                             enforce_sorted=False)
            out, title_states = self.title_rnn(packed, title_states)
            out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out,
                                                                         batch_first=True)  # unpack (back to padded)

            out = out.view(out.size(0), out.size(2), out.size(1))
            out = self.batch_norm_title(out)

            out = out.view(out.size(0) * out.size(2), out.size(1))
            logits = self.dropout(self.title_predictor(out))

            title_prediction = logits.view(-1, self.vocab_size)
        else:
            title_prediction = None

        if output in ["lyrics", "all"]:
            if title_ids is not None:
                title_embed = self.word_embedding(title_ids.squeeze(1))
                title_context_max = torch.max(title_embed, dim=1)[0].unsqueeze(dim=1)
                title_context_over_seq = torch.cat([title_context_max for _ in range(word_embed.size(1))], dim=1)
            else:
                title_context_over_seq = torch.zeros(genre_embed.size(0), genre_embed.size(1), self.word_embedding_size,
                                                     device=self.device)
            input_tensor = torch.cat((artist_embed, genre_embed, keyword_embed, title_context_over_seq, word_embed),
                                     dim=2)

            if len(char_id_length.shape) > 1:
                lengths = char_id_length.squeeze(dim=1).squeeze(dim=1)
            else:
                lengths = char_id_length
            packed = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True,
                                                             enforce_sorted=True)
            out, text_states = self.lyrics_rnn(packed, text_states)
            out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(out,
                                                                         batch_first=True)  # unpack (back to padded)

            out = out.reshape(out.size(0), out.size(2), out.size(1))
            out = self.batch_norm_text(out)

            out = out.view(out.size(0) * out.size(2), out.size(1))

            logits = self.dropout(self.predict_text(out))
            text_prediction = logits.view(-1, self.vocab_size)
        else:
            text_prediction = None

        return title_prediction, title_states, text_prediction, text_states
