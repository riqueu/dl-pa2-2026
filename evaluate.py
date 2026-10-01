import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm

from src.data.mot17 import (
    load_ground_truth,
    get_train_val_split,
    load_seqinfo,
    load_frame_image,
)
from src.detection.detector import MOT17DetectionLoader, TorchvisionDetector
from src.tracking.tracker import BaselineTracker, TemporalTracker
from src.tracking.temporal_model import MotionRNN
from src.metrics import evaluate_sequence
from src.visualization import plot_metrics_over_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation pipeline for MOT17 tracking.")
    parser.add_argument("--mode", type=str, choices=["baseline", "temporal"], default="temporal")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/motion_model.pth")
    parser.add_argument("--cell_type", type=str, choices=["gru", "lstm", "rnn"], default="gru")
    parser.add_argument("--num_layers", type=int, default=1)
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
    if args.subsample < 1:
        raise ValueError("subsample deve ser >= 1.")
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
        model = MotionRNN(input_dim=4, hidden_dim=args.hidden_dim, cell_type=args.cell_type, num_layers=args.num_layers)
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint não encontrado: {args.checkpoint}")
        model.to(device)
        model.eval()

    # 2. Get sequences
    train_seqs, val_seqs = get_train_val_split(args.det_type)
    seqs = train_seqs if args.split == "train" else val_seqs

    per_sequence_results = {}
    summary_metrics = {}

    for seq in seqs:
        print(f"\nEvaluating sequence: {seq}")
        seq_path = os.path.join(args.data_root, "MOT17", "train", seq)
        required_files = ["seqinfo.ini", "gt/gt.txt"]
        if args.det_source == "mot17":
            required_files.append("det/det.txt")
        for required in required_files:
            if not os.path.isfile(os.path.join(seq_path, required)):
                raise FileNotFoundError(os.path.join(seq_path, required))
        seqinfo = load_seqinfo(seq_path)
        gt_tracks, _ = load_ground_truth(seq_path)

        # Setup detector
        if args.det_source == "mot17":
            detector = MOT17DetectionLoader(seq_path)
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
                motion_model=model,
                iou_threshold=args.iou_threshold,
                max_age=args.max_age,
                min_hits=args.min_hits,
                matching=args.matching,
                device=device,
                img_width=seqinfo["imWidth"],
                img_height=seqinfo["imHeight"]
            )

        predicted_tracks = {}
        seq_length = seqinfo["seqLength"]
        if seq_length < 1:
            raise ValueError(f"Sequência vazia: {seq}")
        frames_to_process = list(range(1, seq_length + 1, args.subsample))

        sampled_frames = set(frames_to_process)

        # Run tracking loop
        for frame_id in tqdm(frames_to_process, desc=f"Tracking {seq}"):
            if args.det_source == "torchvision":
                img = load_frame_image(seq_path, frame_id)
                dets = detector.detect(img)
            else:
                # MOT17 detections are usually pre-loaded or queried by frame
                dets = detector.get_detections(frame_id)

            # Update tracker
            active_tracks = tracker.update(dets)
            for track in active_tracks:
                predicted_tracks.setdefault(track.track_id, {})[frame_id] = track.bbox.copy()

        # Evaluate
        seq_metrics = evaluate_sequence(
            {track_id: {f: box for f, box in frames.items() if f in sampled_frames}
             for track_id, frames in gt_tracks.items()
             if any(f in sampled_frames for f in frames)}, predicted_tracks)
        seq_metrics["n_gt_tracks"] = len(gt_tracks)
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

        # Save plot for reproducible review.
        plot_metrics_over_sequences(per_sequence_results, output_path=os.path.join(args.output_dir, "idf1.png"))
        print(f"Plot generated.")


if __name__ == "__main__":
    main()
