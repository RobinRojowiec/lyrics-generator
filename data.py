"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: data.py
Date: 13.04.2020

"""
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


def combine(batch_data, pad_token_id=0, device=None, sort_field=None):
    # "moved_block", "state_a", "state_b", "landmark_blocks", "state_diff",
    # "instruction_text", "instruction_token_ids","instruction_length"
    if sort_field is not None:
        batch_data.sort(key=lambda x: x[sort_field].squeeze(dim=1).item(), reverse=True)

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
        self.special_chars = [
            "<start>",
            "<end>"
        ]

    def tokenize(self, text):
        char_ids = []
        tokens = text.split()
        for index, token in enumerate(tokens):
            # prevent tokenization of special chars
            if token in self.special_chars:
                chars = [token]
            else:
                chars = [c for c in token]

            # if not last token, append space character
            if index < len(tokens) - 1:
                chars.append(" ")

            for char in chars:
                if char not in self.chars2id:
                    self.chars2id[char] = self.last_id
                    self.ids2char[self.last_id] = char
                    self.last_id += 1
                char_ids.append(self.chars2id[char])
        return char_ids


class LabelVocab:
    def __init__(self, pad_label=None):
        self.labels = []
        if pad_label is not None:
            self.labels.append(pad_label)

    def get_id(self, label):
        if label in self.labels:
            return self.labels.index(label)
        else:
            self.labels.append(label)
            return len(self.labels) - 1

    def get_label(self, char_id):
        return self.labels[char_id]

    def get_dict(self):
        return {label: i for label, i in enumerate(self.labels)}


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
        self.artist_labels = LabelVocab('<pad>')
        self.genre_labels = LabelVocab('<pad>')

        self.lyrics_key = "lyrics"
        self.artist_key = "artist"
        self.genre_key = "genre"
        self.device = device
        self.pad_id = 0

    def save_vocabs(self, directory="data/"):
        with open(os.path.join(directory, "id2char.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.tokenizer.ids2char, vocab_file, ensure_ascii=False)

        with open(os.path.join(directory, "char2id.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.tokenizer.chars2id, vocab_file, ensure_ascii=False)

        with open(os.path.join(directory, "genres.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.genre_labels.get_dict(), vocab_file, ensure_ascii=False)

        with open(os.path.join(directory, "artists.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.artist_labels.get_dict(), vocab_file, ensure_ascii=False)


    def get_max_length(self):
        return self.max_text_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        data_row = dict()

        # tokenize chars and prepend start and append end token
        song = row[self.lyrics_key]
        char_ids = self.tokenizer.tokenize(song)[:self.max_text_len]

        if len(char_ids) > 1:
            char_ids_target = char_ids[1:]
            char_ids_input = char_ids[:-1]
        else:
            char_ids_target = char_ids
            char_ids_input = char_ids

        # shifted by 1
        char_id_len = len(char_ids_input)
        data_row["char_id_length"] = Variable(torch.LongTensor([[char_id_len]])).to("cpu")

        if char_id_len < self.max_text_len:
            char_ids_input += [self.pad_id for _ in range(self.max_text_len - char_id_len)]
        data_row["char_id_tensor"] = Variable(torch.LongTensor([char_ids_input])).to(self.device)

        if char_id_len < self.max_text_len:
            char_ids_target += [self.pad_id for _ in range(self.max_text_len - char_id_len)]
        data_row["char_id_target_tensor"] = Variable(torch.LongTensor([char_ids_target])).to(self.device)

        genre_id = self.genre_labels.get_id(row[self.genre_key])
        genre_ids = [genre_id for _ in range(char_id_len)]
        genre_ids_len = len(genre_ids)
        if genre_ids_len < self.max_text_len:
            genre_ids += [self.pad_id for _ in range(self.max_text_len - genre_ids_len)]
        data_row["genre_id"] = Variable(torch.LongTensor([genre_ids])).to(self.device)

        artist_id = self.artist_labels.get_id(row[self.artist_key])
        artist_ids = [artist_id for _ in range(char_id_len)]
        artist_ids_len = len(artist_ids)
        if artist_ids_len < self.max_text_len:
            artist_ids += [self.pad_id for _ in range(self.max_text_len - artist_ids_len)]
        data_row["artist_id"] = Variable(torch.LongTensor([artist_ids])).to(self.device)

        return data_row
