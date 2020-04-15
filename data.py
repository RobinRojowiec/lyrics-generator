"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: data.py
Date: 13.04.2020

"""
from collections import defaultdict

import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


def combine(batch_data, pad_token_id=0, device=None):
    # "moved_block", "state_a", "state_b", "landmark_blocks", "state_diff",
    # "instruction_text", "instruction_token_ids","instruction_length"
    all_data = defaultdict(list)
    for dct in batch_data:
        for key in dct:
            all_data[key].append(dct[key])

    for field in all_data:
        three_dims = len(all_data[field][0].shape) == 3
        max_length = max([tensor.size(1) for tensor in all_data[field]])
        for idx, tensor in enumerate(all_data[field]):
            if three_dims:
                zero_tensor = torch.zeros(1, max_length, tensor.size(2), dtype=tensor.dtype)
                zero_tensor[:, :tensor.size(1), :] = tensor
            else:
                zero_tensor = torch.zeros(1, max_length, dtype=tensor.dtype)
                zero_tensor[:, :tensor.size(1)] = tensor
            all_data[field][idx] = zero_tensor
        all_data[field] = torch.stack(all_data[field], dim=0).to(device)

    return all_data


class CharacterTokenizer:
    def __init__(self):
        self.last_id = 3
        self.chars2id = dict({
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2
        })
        self.ids2char = dict({
            0: "<pad>",
            1: "<start>",
            2: "<end>"
        })


    def tokenize(self, text):
        char_ids = []
        for char in text:
            if char not in self.chars2id:
                self.chars2id[char] = self.last_id
                self.ids2char[self.last_id] = char
                self.last_id += 1
            char_ids.append(self.chars2id[char])
        return char_ids


class LabelVocab:
    def __init__(self):
        self.labels = []

    def get_id(self, label):
        if label in self.labels:
            return self.labels.index(label)
        else:
            self.labels.append(label)
            return len(self.labels) - 1

    def get_label(self, char_id):
        return self.labels[char_id]


class LyricsDataset(Dataset):
    def __init__(self, data_file, limit=0, device="cpu"):
        super(LyricsDataset, self).__init__()

        self.csv_file = data_file
        csv_props = {"header": 0, "sep": ",", "encoding": 'utf8'}
        if limit > 0:
            csv_props["nrows"] = limit
        self.data_frame = pd.read_csv(self.csv_file, **csv_props)
        self.max_text_len = self.data_frame.song.str.len().max()

        self.tokenizer = CharacterTokenizer()
        self.artist_labels = LabelVocab()
        self.genre_labels = LabelVocab()

        self.lyrics_key = "lyrics"
        self.artist_key = "artist"
        self.genre_key = "genre"
        self.device = device
        self.pad_id = 0

    def get_max_length(self):
        return self.max_text_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        data_row = dict()

        # tokenize chars and prepend start and append end token
        song = row[self.lyrics_key]
        char_ids = [1] + self.tokenizer.tokenize(song)[:self.max_text_len - 2] + [2]

        char_id_len = len(char_ids)
        data_row["char_id_length"] = torch.LongTensor([[char_id_len]]).to(self.device)

        if char_id_len < self.max_text_len:
            char_ids += [self.pad_id for _ in range(self.max_text_len - char_id_len)]
        data_row["char_id_tensor"] = Variable(torch.LongTensor([char_ids])).to(self.device)

        genre_id = self.genre_labels.get_id(row[self.genre_key])
        data_row["genre_id"] = Variable(torch.LongTensor([[genre_id for _ in range(self.max_text_len)]])).to(
            self.device)

        artist_id = self.artist_labels.get_id(row[self.artist_key])
        data_row["artist_id"] = Variable(torch.LongTensor([[artist_id for _ in range(self.max_text_len)]])).to(
            self.device)

        return data_row
