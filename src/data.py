import glob
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_length: int, encode) -> None:
        super().__init__()
        self.encode = encode
        self.data = torch.tensor(self.encode(data), dtype=torch.int64)
        self.length = len(data)
        self.seq_length = seq_length

    def __len__(self):
        return self.length - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + 1 + self.seq_length]
        return x, y


class CreateDataset:
    def __init__(self, data: str, seq_length: int, size=None) -> None:
        super().__init__()
        self.chars = sorted(set(data))
        self.stoi = {v: k for k, v in enumerate(self.chars)}
        self.itos = {k: v for k, v in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda t: "".join([self.itos[i] for i in t])

        self.seq_length = seq_length
        if size is not None:
            n = int(size * len(data))
            self._train_data = data[:n]
            self._test_data = data[n:]
        else:
            self._train_data = data
            self._test_data = None

    def train_dataset(self):
        return BaseDataset(self._train_data, self.seq_length, self.encode)

    def test_dataset(self):
        if self._test_data is None:
            return
        return BaseDataset(self._test_data, self.seq_length, self.encode)


def read_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    return text


def preprocess_text(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)
    text = re.sub(r"\d{1,}\n\n", "", text)
    text = re.sub(r"- ", "", text)
    text = re.sub(r"<\w+>((.|\n)*?)</\w+>", r"\1", text)
    text = re.sub(r"[-:–—]", "", text)
    return text


def extract_dialogue(file_path):
    """
    Extracts dialogue from an SRT file and returns it as strings.
    """
    with open(file_path, "r") as f:
        srt = f.read()

    # Split the SRT into individual subtitle blocks
    blocks = srt.strip().split("\n\n")

    # Extract the dialogue from each subtitle block
    dialogue = []
    for block in blocks:
        # Remove any tags or timestamps from the subtitle block
        block = re.sub("<.*?>", "", block)
        block = re.sub("\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", block)
        block = block.split("\n\n")[1:]

        dialogue.append("".join(block))

    return "\n\n".join(dialogue)


def prepare_data(dir):
    dir = Path(dir)
    file_paths = dir.glob("*.srt")
    # dialogues = [extract_dialogue(file_path) for file_path in file_paths]
    # dialogues = "".join([extract_dialogue(file_path) for file_path in file_paths])
    texts = [read_srt(f) for f in file_paths]
    dialogues = [preprocess_text(t) for t in texts]
    dialogues = "".join(dialogues)
    return dialogues
