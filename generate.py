"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: generate.py
Date: 13.04.2020

"""

import json
import os
from collections import defaultdict

import torch
import torch.nn as nn

from model import TextGenerator, LyricsGenerator


def make_id_tensor(id: int):
    return torch.LongTensor([[[id]]])


def _generate(model, artist_id, genre_id, title_ids, title_id_length, id2char_vocab=None, start_id=1, temperature=1.0,
              max_length=500, end_char_id=2,
              states=None):
    next_word = torch.LongTensor([[[start_id]]])
    generated_chars = [id2char_vocab[str(start_id)]]

    char_counter = defaultdict(lambda: 1)
    for _ in range(max_length):
        predicted_word, states = model(artist_id=artist_id, genre_id=genre_id, char_id_tensor=next_word,
                                       char_id_length=torch.LongTensor([1]), title_ids=title_ids,
                                       title_id_length=title_id_length, states=states)
        predicted_word = predicted_word.squeeze(0)
        output_dist = nn.functional.softmax(predicted_word.div(temperature), dim=0).data

        # penalty for repeating chars
        for key in char_counter.keys():
            output_dist[key] = output_dist[key] * 1.0 / char_counter[key] ** 1

        predicted_label = torch.multinomial(output_dist, 1)
        char_id = predicted_label.item()

        # prevent repeating chars to often and reset if not directly repeated
        char_counter[char_id] += 1
        for key in char_counter.keys():
            if key != char_id:
                char_counter[key] = 1

        # end the lyrics if end character appears
        if char_id == end_char_id:
            break

        generated_chars.append(id2char_vocab.get(str(char_id), " "))
        next_word = torch.LongTensor([[[char_id]]])

    text = "".join(generated_chars)
    return text


def generate_song(whole_model: LyricsGenerator, artist_id, genre_id, id2vocab, vocab2id, max_length=100,
                  max_title_length=100):
    title_model = whole_model.get_title_generator()
    generated_title: str = _generate(title_model, id2char_vocab=id2vocab, artist_id=artist_id, genre_id=genre_id,
                                     title_ids=None, title_id_length=None, max_length=max_title_length, temperature=1.0)

    title_ids = torch.LongTensor([[vocab2id[title_char] for title_char in generated_title]])
    title_id_length: int = torch.LongTensor([title_ids.size(1)])

    lyrics_model = whole_model.get_lyrics_generator()
    generated_lyrics: str = _generate(lyrics_model, id2char_vocab=id2vocab, artist_id=artist_id, genre_id=genre_id,
                                      title_ids=title_ids, title_id_length=title_id_length, max_length=max_length)

    return generated_title, generated_lyrics


def format_generated_text(text, replace_special_chars=True, insert_line_breaks=True):
    if replace_special_chars:
        text = text.replace("<start>", "").replace("<end>", "").replace("<pad>", " ").strip()

    if insert_line_breaks:
        formatted = ""
        for char in text:
            if char.isupper():
                formatted += os.linesep
            formatted += char
        return formatted
    return text


if __name__ == '__main__':
    # set parameters
    artist = make_id_tensor(1)
    genre = make_id_tensor(1)
    length = 100
    states = None
    id2char_vocab_file = "data/id2char.vocab"
    char2id_vocab_file = "data/char2id.vocab"

    # load vocab files
    with open(id2char_vocab_file, "r", encoding="utf8") as json_file:
        id2char_vocab = json.load(json_file)

    with open(char2id_vocab_file, "r", encoding="utf8") as json_file:
        char2id_vocab = json.load(json_file)

    # load model
    model = TextGenerator()
    model.load_state_dict(torch.load("data/lstm_model.pt", map_location=torch.device("cpu")))
    model.eval()

    title, lyrics = generate_song(model, artist, genre, id2char_vocab)
    print(lyrics)
