import re
from collections import Counter

with open("data/wiki.train.tokens", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = re.findall(r"\b[a-zA-Z]+\b", text)

# count words
word_counts = Counter(tokens)

# keep top 10k words
vocab_size = 10000
vocab = [w for w, _ in word_counts.most_common(vocab_size)]

# special tokens
special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]

# add numbers 0–1000
numbers = [str(i) for i in range(1001)]

vocab = special_tokens + vocab + numbers
with open("gen/vocab_10k_plus.txt", "w", encoding="utf-8") as f:
    for w in vocab:
        f.write(w + "\n")

print("Final vocab size:", len(vocab))
