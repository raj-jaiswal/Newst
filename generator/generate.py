# --- generator/generate.py ---
"""
Generation script. Keeps generating until it finds <END>.
If <UNK> is produced, it is skipped (never selected).
Prints the softmax vector `y` at each timestep.
"""

if __name__ == '__main__':
    import argparse
    import torch
    import numpy as np
    from pathlib import Path

    from generator.model_utils import load_embeddings_csv, RNNGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='generator/checkpoints/checkpoint_epoch5.pt')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='sample', choices=['greedy', 'sample'],
                        help="Choose 'greedy' (argmax) or 'sample' (probabilistic sampling).")
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=device)
    word2idx = ckpt['word2idx']
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = ckpt['vocab_size']
    emb_dim = ckpt['emb_dim']

    # load original embeddings to initialize embedding matrix similarly
    emb_csv = Path('gen/embeddings_new.csv')
    _w2i, emb_matrix, _emb_dim = load_embeddings_csv(emb_csv)

    model = RNNGenerator(vocab_size=vocab_size, emb_dim=emb_dim, emb_matrix=emb_matrix).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    start_idx = word2idx['<START>']
    end_idx = word2idx['<END>']
    unk_idx = word2idx['<UNK>']

    seq = [start_idx]
    hidden = None
    generated = None

    print('Starting generation...')
    for t in range(args.max_len):
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab)
            last_logits = logits[0, -1, :]  # (vocab,)
            # apply temperature
            probs = torch.softmax(last_logits / max(1e-8, args.temperature), dim=0)
            y = probs.cpu().numpy()

        # print the full y vector
        np.set_printoptions(precision=4, suppress=True)
        print('y =', y)

        if args.mode == 'greedy':
            next_idx = int(np.argmax(y))
        else:  # sampling mode
            y[unk_idx] = 0.0  # mask <UNK>
            if y.sum() == 0:
                next_idx = np.random.randint(0, vocab_size)  # fallback
            else:
                y = y / y.sum()
                next_idx = int(np.random.choice(len(y), p=y))

        print('chosen ->', idx2word.get(next_idx, '<UNK>'))

        if next_idx == end_idx:
            generated = [idx2word.get(i, '<UNK>') for i in seq + [end_idx]]
            break

        seq.append(next_idx)

    if generated is not None:
        # strip start/end/pad
        tokens = [w for w in generated if w not in ('<START>', '<END>', '<PAD>')]
        print('FINAL GENERATED TITLE:', ' '.join(tokens))
    else:
        print('Generation ended without finding <END>.')
