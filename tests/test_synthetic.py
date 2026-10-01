"""Testes unitários para o gerador de vídeos sintéticos."""

import numpy as np
import pytest
from src.data.synthetic import generate_synthetic_video, degrade_detections, create_metric_edge_cases


def test_video_shape():
    """Verifica dimensões corretas do vídeo gerado."""
    frames, gt_tracks, gt_per_frame = generate_synthetic_video(n_frames=10, n_objects=3, img_size=128)
    assert frames.shape == (10, 128, 128, 3)
    assert frames.dtype == np.uint8


def test_track_count():
    """Verifica que o número de tracks GT corresponde a n_objects."""
    frames, gt_tracks, gt_per_frame = generate_synthetic_video(n_frames=10, n_objects=5, img_size=128)
    assert len(gt_tracks) == 5


def test_bbox_within_bounds():
    """Verifica que todos os bboxes estão dentro dos limites da imagem."""
    img_size = 128
    frames, gt_tracks, gt_per_frame = generate_synthetic_video(n_frames=10, n_objects=3, img_size=img_size)
    for track_id, frames_dict in gt_tracks.items():
        for frame_id, bbox in frames_dict.items():
            assert bbox[0] >= 0, f"x1 < 0 for track {track_id}, frame {frame_id}"
            assert bbox[1] >= 0, f"y1 < 0 for track {track_id}, frame {frame_id}"
            assert bbox[2] <= img_size, f"x2 > img_size for track {track_id}, frame {frame_id}"
            assert bbox[3] <= img_size, f"y2 > img_size for track {track_id}, frame {frame_id}"


def test_degradation():
    """Verifica que degrade_detections com drop_rate=0.5 reduz detecções."""
    frames, gt_tracks, gt_per_frame = generate_synthetic_video(n_frames=20, n_objects=10, img_size=128, seed=42)
    degraded = degrade_detections(gt_per_frame, drop_rate=0.5, seed=42)

    total_gt = sum(len(v) for v in gt_per_frame.values())
    total_degraded = sum(len(v) for v in degraded.values())
    # With 50% drop rate, expect roughly half (with some variance)
    assert total_degraded < total_gt, "Degraded should have fewer detections than GT"


def test_edge_cases():
    """Verifica que create_metric_edge_cases retorna 3 tuplas (gt, pred, descr)."""
    cases = create_metric_edge_cases()
    assert len(cases) == 3
    for case in cases:
        assert len(case) == 3  # (gt_tracks, pred_tracks, description)
        gt, pred, desc = case
        assert isinstance(gt, dict)
        assert isinstance(pred, dict)
        assert isinstance(desc, str)
