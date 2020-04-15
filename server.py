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

from generate import generate, make_id_tensor

app = FastAPI()

# load model
model = torch.load("data/lstm_model.pt", map_location=torch.device("cpu"))
model.eval()

# load vocab files
id2char_vocab_file = "data/id2char.vocab"
with open(id2char_vocab_file, "r", encoding="utf8") as json_file:
    id2char_vocab = json.load(json_file)

with open("data/genres.vocab", "r", encoding="utf8") as json_file:
    genres_vocab = json.load(json_file)

with open("data/artists.vocab", "r", encoding="utf8") as json_file:
    artist_vocab = json.load(json_file)


@app.get("/api/info")
def api_info():
    return "Lyrics Generator Model 0.0.1"


@app.get("/api/parameters")
def api_info():
    return {
        "genre": [key for key in genres_vocab.values()][1:],
        "artist": [key for key in artist_vocab.values()][1:]
    }


@app.post("/api/generate", description="Analyzes news facts and returns findings")
def extract_text(artist: int, genre: int, length: int = 1000):
    if artist is not None and genre is not None:
        artist_id = make_id_tensor(artist)
        genre_id = make_id_tensor(genre)
        lyrics = generate(artist_id, genre_id, id2char_vocab, max_length=length)
        return {
            "lyrics": lyrics.replace("<start>", "").replace("<end>", ""),
            "length": {
                "chars": len(lyrics),
                "words": len(lyrics.split())
            }
        }
    raise HTTPException(status_code=400, detail="Invalid request")


app.mount("/", StaticFiles(directory="public"), name="static")

if __name__ == "__main__":
    port = os.getenv("PORT", 8000)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
