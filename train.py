import argparse
import json
import os
import random
import time
from typing import Dict, Any

import numpy as np
import torch

from src.data.mot17 import get_train_val_split, load_ground_truth, load_seqinfo
from src.tracking.temporal_model import MotionRNN, extract_training_sequences, train_motion_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MotionRNN for MOT17")
    parser.add_argument("--cell_type", type=str, default="gru", choices=["gru", "lstm", "rnn"],
                        help="RNN cell type (default: gru)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size (default: 64)")
    parser.add_argument("--input_dim", type=int, default=4, help="Input dimension size (default: 4)")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of RNN layers (default: 1)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--tbptt_len", type=int, default=16, help="Truncated Backpropagation Through Time length (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--det_type", type=str, default="SDP", choices=["SDP", "DPM", "FRCNN"],
                        help="Detector type (default: SDP)")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of MOT17 data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/motion_model.pth",
                        help="Path to save the model checkpoint")
    parser.add_argument("--out", type=str, default="runs/train", help="Output directory for logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    return parser.parse_args()


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    print(f"Device: {args.device} | Detector: {args.det_type} | Cell: {args.cell_type} | Epochs: {args.epochs}")

    train_splits, _ = get_train_val_split(args.det_type)
    all_sequences = []

    print("Loading ground truth tracks...")
    for seq_name in train_splits:
        seq_path = os.path.join(args.data_root, "MOT17", "train", seq_name)
        if not os.path.exists(seq_path):
            print(f"Warning: Sequence path {seq_path} does not exist. Skipping.")
            continue

        seq_info = load_seqinfo(seq_path)
        im_width = seq_info.get("imWidth", 1920)
        im_height = seq_info.get("imHeight", 1080)

        gt_tracks, _ = load_ground_truth(seq_path)

        seqs = extract_training_sequences(
            gt_tracks, min_length=10, img_width=im_width, img_height=im_height)
        all_sequences.extend(seqs)

    if not all_sequences:
        print("No training sequences found. Check your data root and split.")
        raise SystemExit(1)

    print(f"Extracted {len(all_sequences)} sequences for training.")

    # Model definition
    model = MotionRNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cell_type=args.cell_type
    )

    # Train
    print("Starting training...")
    start_time = time.time()
    history = train_motion_model(
        model=model,
        sequences=all_sequences,
        epochs=args.epochs,
        lr=args.lr,
        tbptt_len=args.tbptt_len,
        device=args.device,
        batch_size=args.batch_size
    )
    elapsed = time.time() - start_time

    # Save checkpoint
    torch.save(model.state_dict(), args.checkpoint)

    # Save arguments
    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Save history
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    final_loss = history["loss"][-1] if history["loss"] else float("inf")
    print(f"Training completed in {elapsed / 60:.2f} minutes.")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Logs saved to {args.out}")


if __name__ == "__main__":
    main()
