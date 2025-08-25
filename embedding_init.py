import numpy as np
import csv

with open("gen/vocab_10k_plus.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

embedding_dim = 100  

embeddings = np.random.normal(0, 0.1, (len(vocab), embedding_dim))

with open("gen/embeddings.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["word"] + [f"e{i}" for i in range(embedding_dim)]
    writer.writerow(header)

    for word, vector in zip(vocab, embeddings):
        writer.writerow([word] + list(vector))

print("Saved embeddings.csv with shape:", embeddings.shape)