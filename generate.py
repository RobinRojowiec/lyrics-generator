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


def make_id_tensor(id: int):
    return Variable(torch.LongTensor([[[id]]]))


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
model = torch.load("data/lstm_model.pt", map_location=torch.device("cpu"))
model.eval()


# TODO: add initial text to continue generation, add song title as parameter,
def generate(artist_id, genre_id, id2char_vocab, start_id=1, temperature=1.0, max_length=500, end_char_id=2,
             states=None):
    next_word = Variable(torch.LongTensor([[[start_id]]]))
    generated_chars = [id2char_vocab[str(start_id)]]

    for _ in range(max_length):
        predicted_word, states = model(artist_id=artist_id, genre_id=genre_id, char_id_tensor=next_word,
                                       char_id_length=torch.LongTensor([1]), states=states)
        predicted_word = predicted_word.squeeze(0)
        output_dist = nn.functional.softmax(predicted_word.div(temperature), dim=0).data
        predicted_label = torch.multinomial(output_dist, 1)
        char_id = predicted_label.item()

        # end the lyrics if end character appears
        if char_id == end_char_id:
            break

        generated_chars.append(id2char_vocab[str(char_id)])
        next_word = Variable(torch.LongTensor([[[char_id]]]))

    text = "".join(generated_chars)
    return text


if __name__ == '__main__':
    lyrics = generate(artist, genre, id2char_vocab)
    print(lyrics)
