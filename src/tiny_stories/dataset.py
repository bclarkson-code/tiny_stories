import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from tiny_stories.config import Config
from tiny_stories.data.prepare import Split

config = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(config)


class Dataset(TorchDataset):
    def __init__(
        self,
        split: Split,
        config: Config,
    ):
        """
        Args:
            documents: list[list[str]] - list of documents, each document is a list of token strings
            tokenizer: tokenizer with vocab (or dict mapping tokens to ids)
            context_window: maximum sequence length
            eos_token: end of sequence token string
        """
        self.config = config
        self.split = split
        self.context_window = config.context_window
        self.eot_token_id = config.eot_token_id
        documents = self.load_documents()
        self.samples = self._pack_documents(documents)

    def load_documents(self) -> list[list[int]]:
        match self.split:
            case Split.TRAIN:
                path = Path(__file__).parent / "data" / self.config.train_token_path
            case Split.VALID:
                path = Path(__file__).parent / "data" / self.config.valid_token_path
        logger.info(f"Loading: {self.split.value} dataset")
        with open(path, "rb") as f:
            documents = pickle.load(f)
        return documents

    def _pack_documents(self, documents: list[list[int]]) -> list[list[int]]:
        packed = []
        current_sample = []

        for doc in tqdm(documents, desc="Packing"):
            # add an end of string token to the end of the document
            doc += [self.eot_token_id]

            # If adding this doc would exceed context window length, drop it and pad instead
            if len(current_sample) + len(doc) > self.context_window:
                if current_sample:
                    packed.append(current_sample)
                current_sample = []

            # If single doc is longer than context window, split it
            if len(doc) > self.context_window:
                for i in range(0, len(doc), self.context_window):
                    chunk = doc[i : i + self.context_window]
                    packed.append(chunk)
            else:
                current_sample.extend(doc)

        if current_sample:
            packed.append(current_sample)

        return packed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if len(sample) < self.context_window:
            padding_length = self.context_window - len(sample)
            sample = sample + [self.eot_token_id] * padding_length

        input_ids = torch.tensor(sample[:-1], dtype=torch.long)  # All but last token
        labels = torch.tensor(sample[1:], dtype=torch.long)  # All but first token

        return {"input_ids": input_ids, "labels": labels}


def load_datasets(config: Config) -> tuple[Dataset, Dataset]:
    train_ds = Dataset(split=Split.TRAIN, config=config)
    logger.info(f"Loaded train_ds with {len(train_ds)} batches")
    valid_ds = Dataset(split=Split.VALID, config=config)
    logger.info(f"Loaded valid_ds with {len(valid_ds)} batches")

    return train_ds, valid_ds


def load_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = load_datasets(config)
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,  # dont shuffle validation data so we always eval on the same subset
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    return train_dl, valid_dl


if __name__ == "__main__":
    # sanity check
    config = Config()
    split = Split.VALID
    ds = Dataset(split=split, config=config)
    logger.info(ds[0])
