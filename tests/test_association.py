"""Testes unitários para algoritmos de associação (Hungarian e Greedy matching)."""

import numpy as np
import pytest
from src.tracking.association import hungarian_matching, greedy_matching


def test_hungarian_perfect():
    """Matching perfeito: diagonal alta na matriz de IoU."""
    iou_matrix = np.array([[0.9, 0.1], [0.1, 0.8]])
    matches, unmatched_a, unmatched_b = hungarian_matching(iou_matrix, threshold=0.5)
    assert len(matches) == 2
    match_set = set(matches)
    assert (0, 0) in match_set
    assert (1, 1) in match_set
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0


def test_greedy_perfect():
    """Matching guloso com diagonal alta."""
    iou_matrix = np.array([[0.9, 0.1], [0.1, 0.8]])
    matches, unmatched_a, unmatched_b = greedy_matching(iou_matrix, threshold=0.5)
    assert len(matches) == 2
    match_set = set(matches)
    assert (0, 0) in match_set
    assert (1, 1) in match_set


def test_threshold_filtering():
    """IoU abaixo do threshold → nenhum match."""
    iou_matrix = np.array([[0.2, 0.1], [0.1, 0.3]])
    matches, unmatched_a, unmatched_b = hungarian_matching(iou_matrix, threshold=0.5)
    assert len(matches) == 0
    assert len(unmatched_a) == 2
    assert len(unmatched_b) == 2


def test_unmatched():
    """Mais detecções que tracks → unmatched_cols > 0."""
    iou_matrix = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2]])
    matches, unmatched_a, unmatched_b = hungarian_matching(iou_matrix, threshold=0.5)
    assert len(matches) == 2
    assert len(unmatched_b) == 1


def test_empty():
    """Sem tracks → todos os detections são unmatched."""
    iou_matrix = np.empty((0, 3))
    matches, unmatched_a, unmatched_b = hungarian_matching(iou_matrix, threshold=0.5)
    assert len(matches) == 0
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 3
