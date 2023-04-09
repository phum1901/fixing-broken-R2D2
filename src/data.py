import torch
from torch.utils.data import DataLoader, Dataset, random_split


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
            self.__train_data = data[:n]
            self.__test_data = data[n:]
        else:
            self.__train_data = data
            self.__test_data = None

    def train_dataset(self):
        return BaseDataset(self.__train_data, self.seq_length, self.encode)

    def test_dataset(self):
        if self.__test_data is None:
            return
        return BaseDataset(self.__test_data, self.seq_length, self.encode)
