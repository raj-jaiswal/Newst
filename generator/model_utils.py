#  Common imports, tokenization, embedding loader, dataset and model class.
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<END>"]


def tokenize(text: str):
    text = (text or "").lower()
    text = text.replace("—", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.findall(r"[a-z0-9']+", text)


def load_embeddings_csv(path: Path) -> Tuple[Dict[str,int], np.ndarray, int]:
    """
    Load a CSV of embeddings (word,e0..eN). Returns (word2idx, emb_matrix, emb_dim)
    Preserves the file order as the vocabulary order.
    If SPECIAL_TOKENS exist in file they will be respected; if not they will be inserted at the front
    (PAD first, then UNK, START, END).
    """
    if not path.exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    emb_dim = df.shape[1]

    # preserve order from CSV but ensure special tokens present at the front
    vocab = []
    seen = set()
    for st in SPECIAL_TOKENS:
        if st in df.index and st not in seen:
            vocab.append(st)
            seen.add(st)

    for w in df.index:
        if w not in seen:
            vocab.append(w)
            seen.add(w)

    # ensure all SPECIAL_TOKENS present (if not in file, add them now at front)
    for st in reversed(SPECIAL_TOKENS):
        if st not in seen:
            vocab.insert(0, st)
            seen.add(st)

    word2idx = {w:i for i,w in enumerate(vocab)}

    # build matrix in vocab order
    emb_matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    for w,i in word2idx.items():
        if w in df.index:
            vals = df.loc[w].values.astype(np.float32)
            if vals.shape[0] == emb_dim:
                emb_matrix[i] = vals
            else:
                emb_matrix[i] = np.random.normal(0.0, 0.01, size=(emb_dim,)).astype(np.float32)
        else:
            # random init for tokens not in CSV (rare if we inserted special tokens)
            emb_matrix[i] = np.random.normal(0.0, 0.01, size=(emb_dim,)).astype(np.float32)

    return word2idx, emb_matrix, emb_dim


class TitlesDataset(Dataset):
    """Dataset that handles only title column from a CSV and maps tokens -> indices.
    Unknown words are mapped to <UNK>.
    """
    def __init__(self, csv_path: Path, word2idx: Dict[str,int], max_len: int = 32):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"News CSV not found: {csv_path}")
        self.df = pd.read_csv(self.csv_path)
        # try common title column names
        if 'title' in self.df.columns:
            self.titles = self.df['title'].astype(str).fillna("").tolist()
        else:
            # fallback: take first column
            self.titles = self.df.iloc[:,0].astype(str).fillna("").tolist()

        self.word2idx = word2idx
        self.unk_idx = word2idx.get('<UNK>')
        self.start_idx = word2idx.get('<START>')
        self.end_idx = word2idx.get('<END>')
        self.pad_idx = word2idx.get('<PAD>')
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.titles)

    def _encode(self, text: str) -> List[int]:
        toks = tokenize(text)
        seq = [self.start_idx]
        for t in toks[: (self.max_len - 2)]:
            seq.append(self.word2idx.get(t, self.unk_idx))
        seq.append(self.end_idx)
        # pad
        if len(seq) < self.max_len:
            seq += [self.pad_idx] * (self.max_len - len(seq))
        return seq

    def __getitem__(self, idx: int):
        return torch.tensor(self._encode(self.titles[idx]), dtype=torch.long)


class RNNGenerator(nn.Module):
    """
    Simple RNN (LSTM) next-token model. Hidden size and number of neurons per layer default to
    vocab_size. Output dimension is vocab_size followed by softmax in generation.
    """
    def __init__(self, vocab_size: int, emb_dim: int, emb_matrix: np.ndarray = None,
                 num_layers: int = 1, dropout: float = 0.1, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        # embedding layer initialized from emb_matrix if provided
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.from_numpy(emb_matrix))
        # per your instruction: hidden size = vocab_size (for each layer)
        self.hidden_size = self.vocab_size
        self.num_layers = num_layers
        lstm_dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size,
                    num_layers=self.num_layers, batch_first=True, dropout=lstm_dropout)

        self.output = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        out, _ = self.lstm(emb)  # (batch, seq_len, hidden_size)
        logits = self.output(out)  # (batch, seq_len, vocab_size)
        return logits