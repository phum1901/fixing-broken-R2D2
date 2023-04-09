import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import StarWarsDataset
from src.models import Model

# params
epochs = 100

torch.manual_seed(555)
args = argparse.Namespace
args.batch_size = 32
args.vocab_size = vocab_size
args.seq_length = 8
args.n_embd = 64
args.head_size = 16
args.n_head = 4
