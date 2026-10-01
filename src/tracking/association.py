"""Módulo de associação espacial via IoU para tracking.

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-association
Teste unitário: pytest tests/test_association.py
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Calcula matriz de IoU entre dois conjuntos de caixas [x1, y1, x2, y2] de forma vetorizada.

    Args:
        bboxes_a: Array numpy (N, 4).
        bboxes_b: Array numpy (M, 4).

    Returns:
        Array numpy (N, M) contendo IoU entre cada par.
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    bboxes_a = np.asarray(bboxes_a, dtype=np.float32)
    bboxes_b = np.asarray(bboxes_b, dtype=np.float32)

    # Coordenadas da interseção
    inter_x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    inter_y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    inter_x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    inter_y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Áreas individuais
    area_a = np.maximum(0.0, bboxes_a[:, 2] - bboxes_a[:, 0]) * np.maximum(0.0, bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = np.maximum(0.0, bboxes_b[:, 2] - bboxes_b[:, 0]) * np.maximum(0.0, bboxes_b[:, 3] - bboxes_b[:, 1])

    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.zeros_like(inter_area, dtype=np.float32)
    valid_mask = union_area > 0
    iou[valid_mask] = inter_area[valid_mask] / union_area[valid_mask]

    return iou


def greedy_matching(
    iou_matrix: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Realiza matching guloso associando pares com maior IoU primeiro.

    Args:
        iou_matrix: Matriz (N_tracks, M_detections).
        threshold: Limiar mínimo de IoU para aceitar uma associação.

    Returns:
        matches: Lista de tuplas (track_idx, det_idx).
        unmatched_tracks: Lista de índices de tracks não associados.
        unmatched_dets: Lista de índices de detecções não associadas.
    """
    n_tracks, n_dets = iou_matrix.shape
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    matrix = iou_matrix.copy()
    matches = []
    matched_tracks = set()
    matched_dets = set()

    while True:
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        max_val = matrix[max_idx]

        if max_val < threshold or np.isneginf(max_val):
            break

        trk_idx, det_idx = max_idx
        matches.append((int(trk_idx), int(det_idx)))
        matched_tracks.add(int(trk_idx))
        matched_dets.add(int(det_idx))

        # Suprime linha e coluna já casadas
        matrix[trk_idx, :] = -np.inf
        matrix[:, det_idx] = -np.inf

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


def hungarian_matching(
    iou_matrix: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Realiza matching global ótimo via algoritmo Húngaro (linear_sum_assignment).

    Args:
        iou_matrix: Matriz (N_tracks, M_detections).
        threshold: Limiar mínimo de IoU para aceitar uma associação.

    Returns:
        matches: Lista de tuplas (track_idx, det_idx).
        unmatched_tracks: Lista de índices de tracks não associados.
        unmatched_dets: Lista de índices de detecções não associadas.
    """
    n_tracks, n_dets = iou_matrix.shape
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_tracks = set()
    matched_dets = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets
