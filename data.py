"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: data.py
Date: 13.04.2020

"""
import json
import os
import re
from collections import defaultdict

import pandas as pd
import torch
from nltk import RegexpTokenizer
from torch.autograd import Variable
from torch.utils.data import Dataset


def combine(batch_data, device=None, sort_field=None):
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


class WordTokenizer:
    def __init__(self):
        self.special_tokens = [
            "<pad>",
            "<start>",
            "<end>",
            "<unk>",
            "\n"
        ]
        self.last_id = len(self.special_tokens)
        self.token2id = dict({k: i for i, k in enumerate(self.special_tokens)})
        self.ids2token = dict({i: k for i, k in enumerate(self.special_tokens)})

        self.tokenizer_pattern = re.compile("<start>|<end>|\w+|[^\w\s]+|\n", re.IGNORECASE)
        self.word_tokenizer = RegexpTokenizer(self.tokenizer_pattern)

    def tokenize(self, text):
        tokens = self.word_tokenizer.tokenize(text)
        return tokens

    def tokenize_ids(self, text):
        tokens = self.tokenize(text)
        token_ids = []
        for token in tokens:
            if not token in self.token2id:
                self.token2id[token] = self.last_id
                self.ids2token[self.last_id] = token
                self.last_id += 1
            token_ids.append(self.token2id[token])
        return token_ids

    def store_dicts(self, directory):
        with open(os.path.join(directory, "id2token.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.ids2token, vocab_file, ensure_ascii=False, indent=2)

        with open(os.path.join(directory, "token2id.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.token2id, vocab_file, ensure_ascii=False, indent=2)


class CharacterTokenizer:
    def __init__(self, word_tokenizer):
        self.special_chars = [
            "<pad>",
            "<start>",
            "<end>",
            "\n"
        ]
        self.last_id = len(self.special_chars)
        self.chars2id = dict({k: i for i, k in enumerate(self.special_chars)})
        self.ids2char = dict({i: k for i, k in enumerate(self.special_chars)})

        self.word_tokenizer = word_tokenizer

    def tokenize_ids(self, text):
        chars = self.tokenize(text)

        char_ids = []
        for char in chars:
            if char not in self.chars2id:
                self.chars2id[char] = self.last_id
                self.ids2char[self.last_id] = char
                self.last_id += 1
            char_ids.append(self.chars2id[char])
        return char_ids

    def tokenize(self, text):
        chars = []

        tokens = self.word_tokenizer.tokenize(text)
        for index, token in enumerate(tokens):
            # prevent tokenization of special chars
            if token in self.special_chars:
                chars += [token]
            else:
                chars += [c for c in token]

            # if not last token, append space character
            if index < len(tokens) - 1:
                chars.append(" ")

        return chars


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

        self.tokenizer = WordTokenizer()
        self.artist_labels = LabelVocab('<pad>')
        self.genre_labels = LabelVocab('<pad>')

        self.title_key = "song"
        self.lyrics_key = "lyrics"
        self.artist_key = "artist"
        self.genre_key = "genre"
        self.device = device
        self.pad_id = 0

    def save_vocabs(self, directory="data/"):
        self.tokenizer.store_dicts(directory)

        with open(os.path.join(directory, "genres.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.genre_labels.get_dict(), vocab_file, ensure_ascii=False, indent=2)

        with open(os.path.join(directory, "artists.vocab"), "w+", encoding="utf8") as vocab_file:
            json.dump(self.artist_labels.get_dict(), vocab_file, ensure_ascii=False, indent=2)


    def get_max_length(self):
        return self.max_text_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        data_row = dict()

        # tokenize title
        title = row[self.title_key]
        title_char_ids = self.tokenizer.tokenize_ids(title)[:self.max_text_len]
        title_id_len = len(title_char_ids)

        if title_id_len < self.max_text_len:
            title_char_ids += [self.pad_id for _ in range(self.max_text_len + 1 - title_id_len)]

        if title_id_len > 1:
            title_ids_target = title_char_ids[1:]
            title_char_ids = title_char_ids[:-1]
        else:
            title_ids_target = title_char_ids.clone()
            title_char_ids = title_char_ids.clone()

        data_row["title_ids"] = torch.LongTensor([title_char_ids]).to(self.device)
        data_row["title_ids_target"] = torch.LongTensor([title_ids_target]).to(self.device)
        data_row["title_id_length"] = torch.LongTensor([[title_id_len]]).to("cpu")

        # tokenize chars and prepend start and append end token
        lyrics = row[self.lyrics_key]
        char_ids = self.tokenizer.tokenize_ids(lyrics)[:self.max_text_len]

        if len(char_ids) > 1:
            char_ids_target = char_ids[1:]
            char_ids_input = char_ids[:-1]
        else:
            char_ids_target = char_ids[:]
            char_ids_input = char_ids[:]

        # shifted by 1
        char_id_len = len(char_ids_input)
        data_row["char_id_length"] = torch.LongTensor([[char_id_len]]).to("cpu")

        if char_id_len < self.max_text_len:
            char_ids_input += [self.pad_id for _ in range(self.max_text_len - char_id_len)]
        data_row["char_id_tensor"] = torch.LongTensor([char_ids_input]).to(self.device)

        if char_id_len < self.max_text_len:
            char_ids_target += [self.pad_id for _ in range(self.max_text_len - char_id_len)]
        data_row["char_id_target_tensor"] = torch.LongTensor([char_ids_target]).to(self.device)

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


if __name__ == '__main__':
    char_tok = CharacterTokenizer()
    sent = "<start> This is a sentence. This is the second sentence. <end>"
    print(char_tok.word_tokenizer.tokenize(sent))

    ids = char_tok.tokenize(sent)
    print("\t".join([str(id) for id in ids]))
    print("\t".join([char_tok.ids2char[id] for id in ids]))
