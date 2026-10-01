import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from src.data.mot17 import (
    load_ground_truth,
    load_detections,
    list_sequences,
    get_train_val_split,
    load_seqinfo,
    load_frame_image,
)
from src.detection.detector import MOT17DetectionLoader, TorchvisionDetector
from src.tracking.tracker import BaselineTracker, TemporalTracker
from src.tracking.temporal_model import MotionRNN
from src.metrics import evaluate_sequence
from src.visualization import plot_metrics_over_sequences
from src.nms import nms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation pipeline for MOT17 tracking.")
    parser.add_argument("--mode", type=str, choices=["baseline", "temporal"], default="temporal")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/motion_model.pth")
    parser.add_argument("--cell_type", type=str, choices=["gru", "lstm", "rnn"], default="gru")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--det_type", type=str, choices=["SDP", "DPM", "FRCNN"], default="SDP")
    parser.add_argument("--det_source", type=str, choices=["mot17", "torchvision"], default="mot17")
    parser.add_argument("--split", type=str, choices=["val", "train"], default="val")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--min_hits", type=int, default=3)
    parser.add_argument("--matching", type=str, choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subsample", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PA2 - MOT17 EVALUATION PIPELINE")
    print(f"Mode: {args.mode} | Det Source: {args.det_source} | Det Type: {args.det_type}")
    print(f"Split: {args.split} | Matching: {args.matching} | Subsample: {args.subsample}")
    print("=" * 70)

    # 1. Setup temporal model if needed
    model = None
    if args.mode == "temporal":
        model = MotionRNN(input_dim=4, hidden_dim=args.hidden_dim, cell_type=args.cell_type)
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"[WARNING] Checkpoint not found at {args.checkpoint}. Using untrained weights.")
        model.to(device)
        model.eval()

    # 2. Get sequences
    splits = get_train_val_split()
    seqs = splits.get(args.split, [])
    # Filter by detector type
    seqs = [s for s in seqs if args.det_type in s]

    if not seqs:
        print(f"No sequences found for split '{args.split}' and det_type '{args.det_type}'.")
        return

    per_sequence_results = {}
    summary_metrics = {}

    for seq in seqs:
        print(f"\nEvaluating sequence: {seq}")
        seqinfo = load_seqinfo(args.data_root, seq)
        gt_tracks = load_ground_truth(args.data_root, seq)

        # Setup detector
        if args.det_source == "mot17":
            detector = MOT17DetectionLoader(args.data_root, seq)
        else:
            detector = TorchvisionDetector(device=device)

        # Setup tracker
        if args.mode == "baseline":
            tracker = BaselineTracker(
                iou_threshold=args.iou_threshold,
                max_age=args.max_age,
                min_hits=args.min_hits,
                matching=args.matching
            )
        else:
            tracker = TemporalTracker(
                model=model,
                iou_threshold=args.iou_threshold,
                max_age=args.max_age,
                min_hits=args.min_hits,
                matching=args.matching,
                device=device
            )

        predicted_tracks = []
        seq_length = seqinfo["seqLength"]
        frames_to_process = list(range(1, seq_length + 1, args.subsample))

        # Run tracking loop
        for frame_id in tqdm(frames_to_process, desc=f"Tracking {seq}"):
            if args.det_source == "torchvision":
                img = load_frame_image(args.data_root, seq, frame_id)
                dets = detector(img)
            else:
                # MOT17 detections are usually pre-loaded or queried by frame
                dets = detector.get_frame_detections(frame_id)

            # Update tracker
            active_tracks = tracker.update(dets, frame_id)
            predicted_tracks.extend(active_tracks)

        # Evaluate
        seq_metrics = evaluate_sequence(predicted_tracks, gt_tracks)
        per_sequence_results[seq] = seq_metrics

        # Print per sequence results
        print(f"Results for {seq}:")
        for k, v in seq_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Compute summary metrics
    if per_sequence_results:
        metric_keys = list(per_sequence_results[seqs[0]].keys())
        for k in metric_keys:
            summary_metrics[k] = float(np.mean([res[k] for res in per_sequence_results.values()]))

        print("\n" + "=" * 70)
        print("SUMMARY RESULTS")
        print("=" * 70)
        for k, v in summary_metrics.items():
            print(f"Mean {k}: {v:.4f}")

        # Save metrics to JSON
        per_seq_path = os.path.join(args.output_dir, "per_sequence.json")
        with open(per_seq_path, "w", encoding="utf-8") as f:
            json.dump(per_sequence_results, f, indent=4)

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=4)

        print(f"\nMetrics saved to {args.output_dir}")

        # Plot metrics
        # plot_metrics_over_sequences(per_sequence_results) wait, signature is in the prompt?
        # The prompt says: "Generate plots: - IDF1 per sequence (sorted by difficulty) using src.visualization.plot_metrics_over_sequences"
        plot_metrics_over_sequences(per_sequence_results)
        print(f"Plot generated.")


if __name__ == "__main__":
    main()
