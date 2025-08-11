import os
import json
import argparse
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Keep your module names as-is
from ctc50.ctc_data_ver2 import FixedSentenceJSONDataset, ctc_collate_fixed
from ctc_model import TSFCTCEncoder

# how to run:
# python ctc_train_test.py \
#   --train_root keypoints_aug_50/train \
#   --val_root   keypoints_aug_50/val \
#   --vocab_json vocab_50.json \
#   --epochs 15 --batch_size 8 --lr 1e-4 \
#   --num_trials 10 --eval_sentences 500 \
#   --valhard --print_samples 5

# ---------- Utils: decoding + WER ----------

def greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """
    Collapse repeats and remove blanks.
    logits: [B, T, C] (raw).
    returns: list of token id sequences (no 0s, no repeats).
    """
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # [B, T]
        out = []
        for seq in pred.tolist():
            collapsed, prev = [], None
            for t in seq:
                if t != blank_id and t != prev:
                    collapsed.append(t)
                prev = t
            out.append(collapsed)
        return out

def edit_distance(ref, hyp) -> int:
    R, H = len(ref), len(hyp)
    dp = [[0]*(H+1) for _ in range(R+1)]
    for i in range(R+1): dp[i][0] = i
    for j in range(H+1): dp[0][j] = j
    for i in range(1, R+1):
        for j in range(1, H+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,                 # delete
                dp[i][j-1] + 1,                 # insert
                dp[i-1][j-1] + (ref[i-1] != hyp[j-1])  # substitute if different
            )
    return dp[R][H]

def wer_percent(refs: List[List[int]], hyps: List[List[int]]) -> float:
    err = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    N = sum(len(r) for r in refs)
    return 0.0 if N == 0 else 100.0 * err / N

def _unpack_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
    """Recover per-sample reference sequences from concatenated targets."""
    refs = []
    t = 0
    for L in target_lengths.tolist():
        refs.append(targets[t:t+L].tolist())
        t += L
    return refs

@torch.no_grad()
def _show_samples(batch, model, device, id2word, n=5, blank_id=0):
    """Print a few REF vs HYP pairs for transparency."""
    inputs, input_lengths, targets, target_lengths = batch
    inputs = inputs.to(device)
    input_lengths = input_lengths.to(device)

    logits = model(inputs, input_lengths)  # [B, T, C]
    hyps = greedy_decode(logits, blank_id=blank_id)
    refs = _unpack_targets(targets, target_lengths)

    for i, (r, h) in enumerate(zip(refs, hyps)):
        if i >= n: break
        ref_words = " ".join(id2word.get(tok, f"<{tok}>") for tok in r)
        hyp_words = " ".join(id2word.get(tok, f"<{tok}>") for tok in h)
        print(f"  ex{i+1:02d} | REF: {ref_words}")
        print(f"         HYP: {hyp_words}")

# ---------- Train / Eval ----------

def run_epoch(model, loader, optimizer, device, train=True, blank_id=0):
    model.train(train)
    total_loss, n_samples = 0.0, 0
    crit = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    for batch in loader:
        inputs, input_lengths, targets, target_lengths = batch
        inputs = inputs.to(device)                 # [B, T, 354]
        input_lengths = input_lengths.to(device)   # [B]
        targets = targets.to(device)               # [sum L]
        target_lengths = target_lengths.to(device) # [B]

        logits = model(inputs, input_lengths)      # [B, T, C]
        logp = F.log_softmax(logits, dim=-1)       # [B, T, C]
        logp = logp.transpose(0, 1)                # -> [T, B, C] for CTC

        loss = crit(logp, targets, input_lengths, target_lengths)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = inputs.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)

@torch.no_grad()
def evaluate_wer(model, loader, device, blank_id=0, max_batches=None):
    model.eval()
    refs_all, hyps_all = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        inputs, input_lengths, targets, target_lengths = batch
        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)

        logits = model(inputs, input_lengths)      # [B, T, C]
        hyps = greedy_decode(logits, blank_id=blank_id)

        refs = _unpack_targets(targets, target_lengths)
        refs_all.extend(refs)
        hyps_all.extend(hyps)

    return wer_percent(refs_all, hyps_all)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="e.g., keypoints_aug_50/train")
    ap.add_argument("--val_root",   required=True, help="e.g., keypoints_aug_50/val")
    ap.add_argument("--vocab_json", required=True, help="mapping {word:id} with ids 1..50")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--feature_dim", type=int, default=354)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_trials", type=int, default=10, help="number of WER trials after training")
    ap.add_argument("--eval_sentences", type=int, default=500, help="sentences per trial (approx)")
    ap.add_argument("--print_samples", type=int, default=0, help="print N REF/HYP pairs after each epoch")
    ap.add_argument("--valhard", action="store_true", help="enable harder, more realistic validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reproducibility baseline
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab + inverse map for pretty printing
    with open(args.vocab_json, "r") as f:
        vocab = {k: int(v) for k, v in json.load(f).items()}
    id2word = {i: w for w, i in vocab.items()}

    # Datasets (exactly 10-word sentences, top-50 vocab)
    train_ds = FixedSentenceJSONDataset(
        root_dir=args.train_root,
        vocab_json=args.vocab_json,
        # keep training clean
        # (valhard is for validation realism only)
    )
    val_ds = FixedSentenceJSONDataset(
        root_dir=args.val_root,
        vocab_json=args.vocab_json,
        valhard=args.valhard,  # tougher, more realistic validation if enabled
    )

    num_classes = train_ds.vocab_size + 1  # +blank(0) => 51
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=ctc_collate_fixed, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=ctc_collate_fixed, num_workers=2, pin_memory=True)

    model = TSFCTCEncoder(
        feature_dim=args.feature_dim, num_classes=num_classes,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
        dim_feedforward=args.ff, dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs("models", exist_ok=True)

    # ---- Train ----
    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, opt, device, train=True, blank_id=0)

        # quick WER snapshot each epoch (limit batches for speed)
        val_wer = evaluate_wer(model, val_loader, device, blank_id=0, max_batches=50)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_WER%≈{val_wer:.2f}")

        if args.print_samples > 0:
            # Print a few REF/HYP pairs from the first val batch
            for batch in val_loader:
                _show_samples(batch, model, device, id2word, n=args.print_samples, blank_id=0)
                break

        if val_wer < best_val:
            best_val = val_wer
            ckpt = f"models/TSF_CTC_best_WER{best_val:.2f}.pt"
            torch.save({"model": model.state_dict(), "wer": best_val, "cfg": vars(args)}, ckpt)
            print(f"✅ Saved {ckpt}")

    # ---- Trials: %WER over fresh synthetic sentences, averaged ----
    # Each trial reseeds so we draw different 10-word combos/videos.
    trial_wers = []
    for t in range(args.num_trials):
        seed = args.seed + 1000 + t
        random.seed(seed); torch.manual_seed(seed)

        bs = args.batch_size
        max_batches = max(1, args.eval_sentences // bs)

        wer = evaluate_wer(model, val_loader, device, blank_id=0, max_batches=max_batches)
        trial_wers.append(wer)
        print(f"Trial {t+1:02d} WER% = {wer:.2f}")

    avg_wer = sum(trial_wers) / len(trial_wers)
    print("\n--- Summary ---")
    print("Trials:", ", ".join(f"{w:.2f}%" for w in trial_wers))
    print(f"Average WER% over {len(trial_wers)} trials = {avg_wer:.2f}")
    print("----------------")

if __name__ == "__main__":
    main()
