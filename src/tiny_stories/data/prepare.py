import os
from pathlib import Path
import pickle
from tqdm.contrib.concurrent import process_map
import requests

import tiktoken
from enum import Enum
from logging import getLogger
from tiny_stories.config import Config

logger = getLogger()


class Split(Enum):
    TRAIN = "train"
    VALID = "valid"


def download_raw_dataset(
    split: Split = Split.VALID,
    save_dir: Path = Path(__file__).parent,
    force_download: bool = False,
) -> Path:
    # download the tiny stories train and valid datasets to this folder
    match split:
        case Split.TRAIN:
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            path = save_dir / "train.txt"
        case Split.VALID:
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
            path = save_dir / "valid.txt"

    if path.exists() and not force_download:
        return path

    with open(path, "w", encoding="utf-8") as f:
        data = requests.get(url).text
        f.write(data)
    return path


def tokenise(text: str, split: Split, config: Config, force_tokenise: bool = False):
    """
    Tokenise a string into a list of tokenised documents
    """
    tokeniser = tiktoken.get_encoding(config.tokeniser)

    match split:
        case Split.VALID:
            save_path = Path(__file__).parent / config.valid_token_path
        case Split.TRAIN:
            save_path = Path(__file__).parent / config.train_token_path

    if save_path.exists() and not force_tokenise:
        logger.info(f"{str(save_path)} already exists, not retokenising")
    # The text is in the form of short stories, separated by <|endoftext|> strings so we'll
    # first split into documents and then tokenise each document individually
    # When we're training, we'll try and fit as many documents in out context window as we can
    documents = text.split("<|endoftext|>")

    tokens = process_map(
        tokeniser.encode_ordinary,
        documents,
        max_workers=os.cpu_count(),
        chunksize=1000,
        desc="Tokenising",
    )
    with open(save_path, "wb") as f:
        pickle.dump(tokens, f)


if __name__ == "__main__":
    config = Config()
    split = Split.VALID

    path = download_raw_dataset(split)
    all_words = path.read_text()
    tokenise(text=all_words, split=split, config=config)
