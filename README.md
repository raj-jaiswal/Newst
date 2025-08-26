# Newst: Fake News Generator

This README documents everything we've done so far to prepare a small vocabulary, initialize random embeddings, download the `allthenews` dataset, and produce a single cleaned CSV of **title** + **content** ready for model training.

Data Source: \
https://www.kaggle.com/datasets/bestwater/wikitext-2-v1?resource=download  
https://www.kaggle.com/datasets/cymerunaslam/allthenews/data

#### Fake News Generated
* why donald j is publicly the most thing to buy
* this may be doing a anti price on new people in america
* obama campaign reveals a human electoral opportunity
* one of the most popular view of the us is north korean
* why the bank breaks doors from dubai after scoring territory

---

## Project overview

Goal: prepare a compact vocabulary (10k words + numbers 0–1000), initialize random embeddings for each token, and extract news titles + article content from the *AllTheNews* dataset so you can train an RNN (and learn embeddings during training).

What we have produced:

* `vocab_10k_plus.txt` - vocabulary (special tokens + 10k most common words + `"0".."1000"`).
* `embeddings_init.csv` - random-initialized embeddings for every token in the vocab (CSV: `word,e0,...,eN`).
* Downloaded dataset (via `kagglehub.dataset_download`) and moved files to `data/allthenews/`.
* `data/NewsContents.csv` - combined file containing `title` and `content` from all CSVs.

---

## Files for Embedding Training

Below are the scripts that were used / recommended. Save each as a `.py` file and run them in order.

1. **`vocab.py`**

   * Reads `wiki.train.tokens` (Wikitext-2) or any corpus, tokenizes to alphabetic words, counts frequencies, picks top 10,000.
   * Adds special tokens: `<PAD>, <UNK>, <START>, <END>`.
   * Appends strings `"0"`..`"1000"` as separate tokens.
   * Output: `vocab_10k_plus.txt`.

2. **`init_embeddings.py`**

   * Loads vocabulary file and initializes random vectors (normal with `std=0.1` by default).
   * `embedding_dim` parameter (default 100). Change to 200 if you want bigger vectors.
   * Output: `embeddings.csv` (columns: `word,e0,e1,...`).

3. **`kagglehub_import.py`**

   * Keeps the original two lines the user wanted:

     ```python
     import kagglehub
     path = kagglehub.dataset_download("cymerunaslam/allthenews")
     print("Path to dataset files:", path)
     ```
   * Then moves (copy+remove on Windows-safe approach) the three CSV files from the returned path into `data/allthenews/`.
   * Assumes `path` points to a directory containing `articles1.csv`, `articles2.csv`, `articles3.csv` (not zip).

4. **`allthenews_cleanup.py`**

   * Scans `data/allthenews/*.csv`.
   * Increases Python CSV parser field size limit to avoid `field larger than field limit` errors.
   * Uses `pandas.read_csv(..., engine='python', chunksize=...)` to stream large files.
   * Picks best title/content columns by name and sensible fallbacks (title=3rd column, content=last column when necessary).
   * Normalizes whitespace, drops empty rows, and appends chunk results to `data/all_titles_contents.csv`.

---

## Quick usage (run in project root)

```bash
# 1) Build vocab (assumes wikitext file is available)
python vocab.py

# 2) Initialize embeddings
python embeddings_init.py

# 3) Download dataset and move files into ./data/allthenews
python kaggle_import.py

# 4) Combine CSVs into a single cleaned titles+contents file
python allthenews_cleanup.py
```

> If you prefer the official Kaggle CLI, replace step 3 with the Kaggle API method:

```bash
pip install kaggle
# Put kaggle.json at ~/.kaggle/kaggle.json (set permission 600)
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api=KaggleApi(); api.authenticate(); api.dataset_download_files('cymerunaslam/allthenews', path='data', unzip=True)"
```

---

## Notes

* **Embedding dimension**: 100 is a reasonable starting point for a 11k-token vocab. Try 200 for improved semantic resolution.
* **Numbers**: `"1"` and the word `"one"` are left as separate tokens. If you want mapping from words to digits, add a normalization pass.
* **Windows move errors**: `shutil.move` may fail across filesystems on Windows. Use `shutil.copy2()` + `os.unlink()` to copy-then-delete (safe approach used in scripts).
* **Large CSV fields**: we set `csv.field_size_limit` to a very large value and use `engine='python'` with chunking to avoid parser limits.
* **Memory**: combining huge articles can be memory-heavy - we stream in chunks and append to disk to avoid loading everything at once. Deduplication across files is optional and can be done with a streaming hash filter.

---

## Next steps (suggestions)

* Train embeddings with skip-gram/CBOW (Gensim) on Wikitext-103 or the news corpus, then fine-tune on headlines.
* Build a simple word-level LSTM (PyTorch) where the `nn.Embedding` is initialized from `embeddings_init.csv` and allowed to update during training.
* Experiment with embedding dimensions (100 vs 200) and measure validation perplexity on held-out headlines.

---

## Model Training

We implemented a simple RNN/LSTM-based generator with embeddings initialized from `embeddings_init.csv`.

**Training command (example):**

```bash
python -m generator.train --emb_csv gen/embeddings_new.csv --news_csv data/NewsContents.csv --epochs 5 --batch 64 --device cuda --max_rows 30000
```

Checkpoints are automatically saved under `generator/checkpoints/` as `checkpoint_epochX.pt`.

Progress bars (`tqdm`) show training progress with loss updates per batch.

---

## Text Generation

Once a checkpoint is trained, you can generate fake news titles using:

```bash
python -m generator.generate   --ckpt generator/checkpoints/checkpoint_epoch5.pt   --device cuda   --mode sample   --max_len 50   --temperature 1.0
```

Options:

* `--mode greedy` → always picks the highest-probability token (argmax).
* `--mode sample` → samples probabilistically (avoids `<UNK>` tokens).
* `--temperature` → controls randomness. Higher values (e.g., 1.5) make text more diverse; lower values (e.g., 0.7) make it more conservative.

Generation stops automatically when `<END>` token is produced, and strips `<START>`, `<END>`, and `<PAD>` tokens from the final output.
