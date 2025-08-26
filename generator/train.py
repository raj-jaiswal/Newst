"""
Training script that uses only titles and the provided embeddings CSV.
Save checkpoints to generator/checkpoints/.
"""
if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import time
    import torch
    from torch.utils.data import DataLoader, Subset
    import torch.optim as optim
    import torch.nn as nn
    from tqdm.auto import tqdm

    from generator.model_utils import load_embeddings_csv, TitlesDataset, RNNGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_csv', type=str, default='gen/embeddings_new.csv')
    parser.add_argument('--news_csv', type=str, default='data/NewsData.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_rows', type=int, default=30000)
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading embeddings...')
    word2idx, emb_matrix, emb_dim = load_embeddings_csv(Path(args.emb_csv))
    vocab_size = len(word2idx)
    print(f'Vocab size: {vocab_size}, emb_dim: {emb_dim}')

    dataset = TitlesDataset(Path(args.news_csv), word2idx, max_len=args.max_len)
    # Restrict to first 30k rows only
    dataset = Subset(dataset, range(args.max_rows))

    def collate_fn(batch):
        x = torch.stack(batch, dim=0)
        return x[:, :-1], x[:, 1:]

    model = RNNGenerator(vocab_size=vocab_size, emb_dim=emb_dim, emb_matrix=emb_matrix,
                         num_layers=args.num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])

    ckpt_dir = Path('generator/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = 5000
    num_chunks = len(dataset) // CHUNK_SIZE

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen_batches = 0

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk = Subset(dataset, range(start, end))
            dl = DataLoader(chunk, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

            loop = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs} [chunk {chunk_idx+1}/{num_chunks}]", unit="batch", leave=True)
            for i, (inp, target) in enumerate(loop, start=1):
                inp, target = inp.to(device), target.to(device)
                optimizer.zero_grad()
                logits = model(inp)  # (batch, seq_len, vocab)
                loss = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                seen_batches += 1
                avg_loss = running_loss / seen_batches

                loop.set_postfix({'loss': f"{loss.item():.4f}", 'avg_loss': f"{avg_loss:.4f}"})

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished — avg_loss {avg_loss:.6f} — time {epoch_time:.1f}s")

        # save checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'word2idx': word2idx,
            'emb_dim': emb_dim,
            'vocab_size': vocab_size,
            'epoch': epoch,
        }
        torch.save(ckpt, ckpt_dir / f'checkpoint_epoch{epoch}.pt')
        print(f'Saved checkpoint_epoch{epoch}.pt')
