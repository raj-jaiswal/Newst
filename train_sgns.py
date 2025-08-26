import csv
import re
import random
import time
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ---------------------- Hyperparameters ----------------------
EMBED_DIM = 100         # overwritten if gen/embeddings.csv present
WINDOW_SIZE = 4
NEGATIVE_K = 5
BATCH_SIZE = 512
EPOCHS = 5
LR = 0.05
MIN_COUNT = 1
SAVE_PATH = Path("gen/embeddings_new.csv")   # updated output
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# chunking
MAX_ROWS = 30_000
CHUNK_ROWS = 5_000

# DataLoader workers: 0 on Windows to avoid multiprocessing pickling problems
DL_NUM_WORKERS = 0

# special tokens & embedding file path
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<START>", "<END>"]
EMB_PATH = Path("gen/embeddings.csv")
# -------------------------------------------------------------

print("DEVICE:", DEVICE)

def tokenize(text: str):
    text = (text or "").lower()
    text = text.replace("—", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.findall(r"[a-z0-9']+", text)

def tokenize_row_text(row):
    content = ""
    if "content" in row and pd.notna(row["content"]):
        content += str(row["content"]) + " "
    if "title" in row and pd.notna(row["title"]):
        content += str(row["title"])
    return tokenize(content)

class SGNSDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

NEG_SAMPLER = None  # set in main()

def collate_fn_top(batch):
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negs = NEG_SAMPLER(len(batch), NEGATIVE_K)  # numpy array (batch, neg_k)
    negs = torch.tensor(negs, dtype=torch.long)
    return centers, contexts, negs

class NegativeSampler:
    def __init__(self, counts_array):
        pow_counts = np.array(counts_array, dtype=np.float64) ** 0.75
        # handle all-zero case
        if pow_counts.sum() == 0:
            self.probs = np.ones_like(pow_counts, dtype=np.float64) / len(pow_counts)
        else:
            self.probs = (pow_counts / pow_counts.sum()).astype(np.float64)
    def __call__(self, batch_size, neg_k):
        return np.random.choice(len(self.probs), size=(batch_size, neg_k), p=self.probs)

class SGNSModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, init_in_emb=None):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        if init_in_emb is not None:
            with torch.no_grad():
                self.in_embed.weight.copy_(torch.from_numpy(init_in_emb))
        nn.init.normal_(self.out_embed.weight, mean=0.0, std=0.01)
    def forward(self, centers, pos_contexts, neg_contexts):
        v = self.in_embed(centers)                 # (batch, dim)
        u_pos = self.out_embed(pos_contexts)       # (batch, dim)
        u_neg = self.out_embed(neg_contexts)       # (batch, neg_k, dim)
        pos_score = torch.sum(v * u_pos, dim=1)    # (batch,)
        pos_loss = F.logsigmoid(pos_score)         # (batch,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (batch, neg_k)
        neg_loss = F.logsigmoid(-neg_score).sum(1)              # (batch,)
        loss = - (pos_loss + neg_loss).mean()
        return loss

def save_embeddings(out_path: Path, idx2word, emb_numpy):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["word"] + [f"e{i}" for i in range(emb_numpy.shape[1])]
        writer.writerow(header)
        for i, w in idx2word.items():
            row = [w] + emb_numpy[i].astype(float).tolist()
            writer.writerow(row)

def build_vocab_counts_from_csv(csv_path: Path, max_rows=MAX_ROWS, chunk_rows=CHUNK_ROWS):
    counts = Counter()
    rows_seen = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, iterator=True):
        for _, row in chunk.iterrows():
            if rows_seen >= max_rows:
                break
            toks = tokenize_row_text(row)
            if toks:
                counts.update(toks)
            rows_seen += 1
        if rows_seen >= max_rows:
            break
    return counts, rows_seen

def generate_pairs_from_chunk_rows(chunk_df, word2idx, window_size=WINDOW_SIZE):
    pairs = []
    unk_idx = word2idx.get('<UNK>')
    if unk_idx is None:
        # If <UNK> isn't present, skip unknown words (safer fallback)
        for _, row in chunk_df.iterrows():
            toks = tokenize_row_text(row)
            L = len(toks)
            for i, center in enumerate(toks):
                if center not in word2idx:
                    continue
                center_idx = word2idx[center]
                cur_win = random.randint(1, window_size)
                start = max(0, i - cur_win)
                end = min(L, i + cur_win + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = toks[j]
                    if context not in word2idx:
                        continue
                    pairs.append((center_idx, word2idx[context]))
    else:
        # Map unknown tokens to <UNK> so we train its vector
        for _, row in chunk_df.iterrows():
            toks = tokenize_row_text(row)
            L = len(toks)
            for i, center in enumerate(toks):
                center_idx = word2idx.get(center, unk_idx)
                cur_win = random.randint(1, window_size)
                start = max(0, i - cur_win)
                end = min(L, i + cur_win + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = toks[j]
                    context_idx = word2idx.get(context, unk_idx)
                    pairs.append((center_idx, context_idx))
    return pairs

def load_embeddings_csv(emb_path: Path):
    """
    Load embeddings.csv (word x e0..eN). Return:
      - df: pandas DataFrame indexed by word (if file exists), otherwise None
      - embed_dim: number of columns (int) or None
    """
    if not emb_path.exists():
        return None, None
    df = pd.read_csv(emb_path, index_col=0)
    # strip index whitespace and convert to str
    df.index = df.index.map(lambda x: str(x).strip())
    embed_dim = df.shape[1]
    return df, embed_dim

def main():
    global NEG_SAMPLER, EMBED_DIM

    news_path = Path("data/NewsContents.csv")
    assert news_path.exists(), f"Expected news CSV at {news_path} (update path if different)"

    print(f"Counting tokens from first {MAX_ROWS} rows (chunks of {CHUNK_ROWS}) ...")
    t0 = time.time()
    counts, rows_loaded = build_vocab_counts_from_csv(news_path, max_rows=MAX_ROWS, chunk_rows=CHUNK_ROWS)
    print(f"Rows processed for vocab: {rows_loaded}, unique tokens seen: {len(counts)} (took {time.time()-t0:.1f}s)")

    # Dataset vocab (apply MIN_COUNT) -- only used to compute unk-count; we WILL NOT add these words to final vocab
    dataset_vocab = sorted([w for w,c in counts.items() if c >= MIN_COUNT])
    print(f"Dataset unique tokens (after min_count filter): {len(dataset_vocab)} -- these will be collapsed to <UNK> if not present in embeddings.csv")

    # Load initial embeddings if present
    emb_df, emb_dim = load_embeddings_csv(EMB_PATH)
    if emb_df is not None:
        print(f"Found embeddings file at {EMB_PATH} (embed_dim={emb_dim})")
        EMBED_DIM = emb_dim
        # preserve the order from CSV for words that appear in it
        emb_words_ordered = list(emb_df.index)
    else:
        print("No embeddings CSV found at gen/embeddings.csv - will random init for all words.")
        emb_words_ordered = []

    # Build final vocabulary:
    # We will *only* use words present in embeddings.csv (plus special tokens placed first).
    final_vocab = []
    seen = set()

    for st in SPECIAL_TOKENS:
        if st not in seen:
            final_vocab.append(st)
            seen.add(st)

    # add embedding-file words (preserve order)
    for w in emb_words_ordered:
        if w not in seen:
            final_vocab.append(w)
            seen.add(w)

    if "<UNK>" not in seen:
        final_vocab.insert(1, "<UNK>")  # after <PAD>
        seen.add("<UNK>")

    # final mapping
    word2idx = {w:i for i,w in enumerate(final_vocab)}
    idx2word = {i:w for w,i in word2idx.items()}
    vocab_size = len(final_vocab)
    print(f"Final vocab size (specials + emb-file-words only): {vocab_size}")

    # Prepare initial input embedding matrix of shape (vocab_size, EMBED_DIM)
    rng = np.random.default_rng()
    init_emb = np.empty((vocab_size, EMBED_DIM), dtype=np.float32)

    if emb_df is not None:
        matched = 0
        for i,w in idx2word.items():
            if w in emb_df.index:
                vals = emb_df.loc[w].values.astype(np.float32)
                if vals.shape[0] == EMBED_DIM:
                    init_emb[i] = vals
                    matched += 1
                else:
                    # dimension mismatch: random init
                    init_emb[i] = rng.normal(0.0, 0.01, size=(EMBED_DIM,)).astype(np.float32)
            else:
                # word not in embeddings.csv
                if w == "<PAD>":
                    init_emb[i] = np.zeros((EMBED_DIM,), dtype=np.float32)
                else:
                    init_emb[i] = rng.normal(0.0, 0.01, size=(EMBED_DIM,)).astype(np.float32)
        print(f"Initialized embedding matrix (vocab x dim): {init_emb.shape}, matched words from CSV: {matched}")
    else:
        for i,w in idx2word.items():
            if w == "<PAD>":
                init_emb[i] = np.zeros((EMBED_DIM,), dtype=np.float32)
            else:
                init_emb[i] = rng.normal(0.0, 0.01, size=(EMBED_DIM,)).astype(np.float32)
        print(f"Randomly initialized embedding matrix (vocab x dim): {init_emb.shape}")


    set_final = set(final_vocab)
    unk_count = sum(c for w,c in counts.items() if w not in set_final)
    # also include any counts for a literal '<UNK>' token in the data
    unk_count += counts.get('<UNK>', 0)

    counts_array = []
    for i in range(vocab_size):
        w = idx2word[i]
        if w == '<UNK>':
            counts_array.append(int(unk_count))
        else:
            counts_array.append(int(counts.get(w, 0)))

    NEG_SAMPLER = NegativeSampler(counts_array)

    # model & optimizer
    model = SGNSModel(vocab_size=vocab_size, embed_dim=EMBED_DIM, init_in_emb=init_emb).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # training (streaming chunks)
    print(f"Training: epochs={EPOCHS}, rows_per_chunk={CHUNK_ROWS}, max_rows={MAX_ROWS}")
    for epoch in range(1, EPOCHS + 1):
        print(f"=== Epoch {epoch}/{EPOCHS} ===")
        rows_seen = 0
        chunk_iter = pd.read_csv(news_path, chunksize=CHUNK_ROWS, iterator=True)
        chunk_index = 0
        for chunk_df in chunk_iter:
            if rows_seen >= MAX_ROWS:
                break
            if rows_seen + len(chunk_df) > MAX_ROWS:
                needed = MAX_ROWS - rows_seen
                chunk_df = chunk_df.iloc[:needed]
            rows_seen += len(chunk_df)
            chunk_index += 1

            pairs = generate_pairs_from_chunk_rows(chunk_df, word2idx, window_size=WINDOW_SIZE)
            if not pairs:
                continue

            dataset = SGNSDataset(pairs)
            dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn_top, drop_last=False, num_workers=DL_NUM_WORKERS)

            loop = tqdm(dl, desc=f"Epoch {epoch} chunk {chunk_index}", unit="batch", leave=False)
            running_loss = 0.0
            for batch_idx, (centers, contexts, negs) in enumerate(loop, 1):
                centers = centers.to(DEVICE)
                contexts = contexts.to(DEVICE)
                negs = negs.to(DEVICE)

                optimizer.zero_grad()
                loss = model(centers, contexts, negs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix({"batch": batch_idx, "loss": f"{loss.item():.4f}"})

                if batch_idx % 200 == 0:
                    avg = running_loss / 200
                    tqdm.write(f"Epoch {epoch} chunk {chunk_index} batch {batch_idx} avg_loss {avg:.6f}")
                    running_loss = 0.0

        # save embeddings at the end of each epoch
        trained_in_emb = model.in_embed.weight.detach().cpu().numpy()
        epoch_out = SAVE_PATH.with_name(f"embeddings_new_epoch{epoch}.csv")
        save_embeddings(epoch_out, idx2word, trained_in_emb)
        print(f"Saved embeddings after epoch {epoch} -> {epoch_out}")

    # final save (last epoch) for consistency
    trained_in_emb = model.in_embed.weight.detach().cpu().numpy()
    save_embeddings(SAVE_PATH, idx2word, trained_in_emb)
    print(f"Saved final updated embeddings to {SAVE_PATH}")

if __name__ == "__main__":
    main()
