"""Módulo de associação espacial via IoU para tracking.

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-association
Teste unitário: pytest tests/test_association.py
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Calcula matriz de IoU entre dois conjuntos de caixas [x1, y1, x2, y2].

    Args:
        bboxes_a: Array numpy (N, 4).
        bboxes_b: Array numpy (M, 4).

    Returns:
        Array numpy (N, M) contendo IoU entre cada par.
    """
    raise NotImplementedError("Henrique: implementar cálculo da matriz de IoU.")


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
    raise NotImplementedError("Henrique: implementar casamento guloso (greedy).")


def hungarian_matching(
    iou_matrix: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Realiza matching global ótimo via algoritmo Húngaro (linear_sum_assignment).

    Dica:
        O algoritmo minimiza custo. Crie cost_matrix = 1.0 - iou_matrix.
        Descarte casamentos cujo IoU original for menor que o threshold.

    Args:
        iou_matrix: Matriz (N_tracks, M_detections).
        threshold: Limiar mínimo de IoU para aceitar uma associação.

    Returns:
        matches: Lista de tuplas (track_idx, det_idx).
        unmatched_tracks: Lista de índices de tracks não associados.
        unmatched_dets: Lista de índices de detecções não associadas.
    """
    raise NotImplementedError("Henrique: implementar casamento ótimo via algoritmo Húngaro.")
