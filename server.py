"""

IDE: PyCharm
Project: lyrics-generator
Author: Robin
Filename: server.py
Date: 15.04.2020

"""
import json
import os

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from generate import make_id_tensor, format_generated_text, generate_song
from model import LyricsGenerator

app = FastAPI()

# load model
model = LyricsGenerator()
model.load_state_dict(torch.load("data/lstm_model.pt", map_location=torch.device("cpu")))
model.eval()

# load vocab and id files
id2char_vocab_file = "data/id2char.vocab"
with open(id2char_vocab_file, "r", encoding="utf8") as json_file:
    id2char_vocab = json.load(json_file)

char2id_vocab_file = "data/char2id.vocab"
with open(char2id_vocab_file, "r", encoding="utf8") as json_file:
    char2id_vocab = json.load(json_file)

with open("data/genres.vocab", "r", encoding="utf8") as json_file:
    genres_vocab = json.load(json_file)

with open("data/artists.vocab", "r", encoding="utf8") as json_file:
    artist_vocab = json.load(json_file)


@app.get("/api/info")
def api_info():
    return "Lyrigen l 0.0.1"


@app.get("/api/parameters", description="Retrieves all valid parameters for lyrics generation")
def api_info():
    return {
        "genre": [key for key in genres_vocab.values()][1:],
        "artist": [key for key in artist_vocab.values()][1:]
    }


@app.post("/api/generate", description="Generates song lyrics")
def extract_text(artist: int, genre: int, max_length: int = 1000, insert_line_breaks: bool = False):
    if artist is not None and genre is not None:
        artist_id = make_id_tensor(artist)
        genre_id = make_id_tensor(genre)
        title, lyrics = generate_song(model, artist_id, genre_id, id2char_vocab, char2id_vocab, max_length=max_length)
        title = format_generated_text(title, insert_line_breaks=insert_line_breaks)
        text = format_generated_text(lyrics, insert_line_breaks=insert_line_breaks)
        return {
            "title": title,
            "text": text,
            "length": {
                "chars": len(text),
                "words": len(text.split())
            }
        }
    raise HTTPException(status_code=400, detail="Invalid request")


app.mount("/", StaticFiles(directory="frontend/dist", html="index.html"), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
