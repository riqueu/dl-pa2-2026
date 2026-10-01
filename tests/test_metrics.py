"""Testes unitários para métricas MOT autorais (IDF1, ID switches, fragmentations)."""

import numpy as np
import pytest
from src.metrics import (
    compute_idf1, count_id_switches, count_fragmentations,
    count_unique_id_error, compute_bbox_iou, compute_iou_matrix,
    evaluate_sequence,
)


def test_perfect_prediction():
    """Predição idêntica ao GT → IDF1 ≈ 1.0, ID switches = 0."""
    gt_tracks = {
        1: {1: np.array([10, 10, 20, 20]), 2: np.array([12, 12, 22, 22])},
        2: {1: np.array([30, 30, 40, 40]), 2: np.array([32, 32, 42, 42])},
    }
    pred_tracks = {
        1: {1: np.array([10, 10, 20, 20]), 2: np.array([12, 12, 22, 22])},
        2: {1: np.array([30, 30, 40, 40]), 2: np.array([32, 32, 42, 42])},
    }
    idf1 = compute_idf1(gt_tracks, pred_tracks)
    switches = count_id_switches(gt_tracks, pred_tracks)
    frags = count_fragmentations(gt_tracks, pred_tracks)
    assert idf1 > 0.95, f"Expected IDF1 ≈ 1.0, got {idf1}"
    assert switches == 0
    assert frags == 0


def test_swapped_ids():
    """Dois tracks com IDs invertidos no frame 11 → ID switches = 2."""
    gt_tracks = {
        1: {i: np.array([10.0, 10, 20, 20]) for i in range(1, 21)},
        2: {i: np.array([30.0, 30, 40, 40]) for i in range(1, 21)},
    }
    pred_tracks = {}
    # First 10 frames: correct mapping
    pred_tracks[1] = {i: np.array([10.0, 10, 20, 20]) for i in range(1, 11)}
    pred_tracks[2] = {i: np.array([30.0, 30, 40, 40]) for i in range(1, 11)}
    # Frames 11-20: swapped
    pred_tracks[1].update({i: np.array([30.0, 30, 40, 40]) for i in range(11, 21)})
    pred_tracks[2].update({i: np.array([10.0, 10, 20, 20]) for i in range(11, 21)})

    switches = count_id_switches(gt_tracks, pred_tracks)
    assert switches >= 2, f"Expected at least 2 ID switches, got {switches}"


def test_split_tracks():
    """Um GT track com gap entre pred tracks → fragmentations ≥ 1."""
    gt_tracks = {
        1: {i: np.array([10.0, 10, 20, 20]) for i in range(1, 21)},
    }
    # Pred track 1 covers frames 1-8, pred track 2 covers frames 12-20
    # Gap at frames 9-11 → GT track goes from tracked to untracked → fragmentation
    pred_tracks = {
        1: {i: np.array([10.0, 10, 20, 20]) for i in range(1, 9)},
        2: {i: np.array([10.0, 10, 20, 20]) for i in range(12, 21)},
    }
    frags = count_fragmentations(gt_tracks, pred_tracks)
    assert frags >= 1, f"Expected fragmentations >= 1, got {frags}"


def test_empty_predictions():
    """Sem predições → IDF1 = 0."""
    gt_tracks = {
        1: {1: np.array([10, 10, 20, 20])},
    }
    pred_tracks = {}
    idf1 = compute_idf1(gt_tracks, pred_tracks)
    assert idf1 == 0.0


def test_compute_bbox_iou():
    """Testa IoU com boxes idênticos, disjuntos e parcialmente sobrepostos."""
    # Idênticos → IoU = 1.0
    assert compute_bbox_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == pytest.approx(1.0)
    # Disjuntos → IoU = 0.0
    assert compute_bbox_iou(np.array([0, 0, 10, 10]), np.array([10, 10, 20, 20])) == 0.0
    # Parcialmente sobrepostos
    iou = compute_bbox_iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
    assert 0 < iou < 1


def test_compute_iou_matrix():
    """Testa computação vetorizada de matriz de IoU."""
    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    boxes_b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    iou_mat = compute_iou_matrix(boxes_a, boxes_b)
    assert iou_mat.shape == (2, 2)
    assert iou_mat[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert iou_mat[1, 0] == 0.0


def test_evaluate_sequence():
    """Testa a função integrada de avaliação de sequência."""
    gt_tracks = {
        1: {1: np.array([10, 10, 20, 20]), 2: np.array([12, 12, 22, 22])},
    }
    pred_tracks = {
        1: {1: np.array([10, 10, 20, 20]), 2: np.array([12, 12, 22, 22])},
    }
    result = evaluate_sequence(gt_tracks, pred_tracks)
    assert 'idf1' in result
    assert 'id_switches' in result
    assert 'fragmentations' in result
    assert result['idf1'] > 0.9


def test_unique_id_error():
    """Testa contagem de erro de IDs únicos."""
    gt_tracks = {1: {1: np.array([0, 0, 1, 1])}, 2: {1: np.array([2, 2, 3, 3])}}
    pred_tracks = {1: {1: np.array([0, 0, 1, 1])}}
    assert count_unique_id_error(gt_tracks, pred_tracks) == 1
