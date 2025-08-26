
# --- generator/generate.py ---
"""
Generation script. Keeps generating until it finds <END>. If an <UNK> is produced at any step,
it restarts generation (as you requested). It prints the softmax vector `y` at each timestep.
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
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=device)
    word2idx = ckpt['word2idx']
    idx2word = {i:w for w,i in word2idx.items()}
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

    max_attempts = 100
    attempt = 0
    generated = None

    while attempt < max_attempts:
        attempt += 1
        seq = [start_idx]
        hidden = None
        ok = True
        print(f'Generation attempt {attempt}...')
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

            # choose next token via argmax (greedy). If you prefer sampling, replace with torch.multinomial
            next_idx = int(torch.argmax(probs).item())
            print('chosen ->', idx2word.get(next_idx, '<UNK>'))

            if next_idx == unk_idx:
                print('Generated <UNK> — restarting generation (per rules).')
                ok = False
                break
            seq.append(next_idx)
            if next_idx == end_idx:
                # success
                generated = [idx2word.get(i, '<UNK>') for i in seq]
                break

        if ok and generated is not None:
            # strip start/end
            tokens = [w for w in generated if w not in ('<START>', '<END>', '<PAD>')]
            print('FINAL GENERATED TITLE:', ' '.join(tokens))
            break
        else:
            print('Attempt failed; trying again...')

    if generated is None:
        print('Failed to generate a title without <UNK> in', max_attempts, 'attempts.')
