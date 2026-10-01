"""Módulo de métricas de tracking MOT (autoral).

Regras do PA2:
- Proibido o uso de bibliotecas prontas como motmetrics ou TrackEval.
- As métricas IDF1, trocas de identidade (ID switches) e fragmentações devem ser
  implementadas do zero.

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-association
Teste unitário: pytest tests/test_metrics.py
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calcula a Interseção sobre União (IoU) entre dois bounding boxes [x1, y1, x2, y2].

    Args:
        bbox1: Array float ou int no formato [x1, y1, x2, y2].
        bbox2: Array float ou int no formato [x1, y1, x2, y2].

    Returns:
        float: Valor de IoU em [0.0, 1.0].
    """
    x1 = max(float(bbox1[0]), float(bbox2[0]))
    y1 = max(float(bbox1[1]), float(bbox2[1]))
    x2 = min(float(bbox1[2]), float(bbox2[2]))
    y2 = min(float(bbox1[3]), float(bbox2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area1 = max(0.0, float(bbox1[2]) - float(bbox1[0])) * max(0.0, float(bbox1[3]) - float(bbox1[1]))
    area2 = max(0.0, float(bbox2[2]) - float(bbox2[0])) * max(0.0, float(bbox2[3]) - float(bbox2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Calcula a matriz de IoU entre duas listas de bboxes de forma vetorizada.

    Args:
        bboxes1: Array (N, 4) no formato [x1, y1, x2, y2].
        bboxes2: Array (M, 4) no formato [x1, y1, x2, y2].

    Returns:
        Array numpy (N, M) com o IoU par a par.
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    b1 = np.asarray(bboxes1, dtype=np.float32)
    b2 = np.asarray(bboxes2, dtype=np.float32)

    inter_x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    iou = np.zeros_like(inter_area, dtype=np.float32)
    valid_mask = union_area > 0
    iou[valid_mask] = inter_area[valid_mask] / union_area[valid_mask]

    return iou


def compute_idf1(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> float:
    """Calcula a métrica IDF1 (ID F1-score) global da sequência.

    Definição:
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN) = 2 * IDTP / (N_pred + N_gt)
    """
    gt_ids = list(gt_tracks.keys())
    pred_ids = list(pred_tracks.keys())

    total_gt = sum(len(frames) for frames in gt_tracks.values())
    total_pred = sum(len(frames) for frames in pred_tracks.values())

    if total_gt == 0 or total_pred == 0:
        return 0.0

    # Constrói matriz de casamento global: número de frames onde IoU >= threshold
    weights = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for i, g_id in enumerate(gt_ids):
        g_frames = gt_tracks[g_id]
        for j, p_id in enumerate(pred_ids):
            p_frames = pred_tracks[p_id]
            common_frames = set(g_frames.keys()) & set(p_frames.keys())
            matches_count = 0
            for f in common_frames:
                if compute_bbox_iou(g_frames[f], p_frames[f]) >= iou_threshold:
                    matches_count += 1
            weights[i, j] = matches_count

    # Bipartite matching global ótimo (maximiza sobreposições)
    row_ind, col_ind = linear_sum_assignment(-weights)
    idtp = float(sum(weights[r, c] for r, c in zip(row_ind, col_ind)))

    denominator = float(total_pred + total_gt)
    if denominator == 0.0:
        return 0.0

    return float(2.0 * idtp / denominator)


def count_id_switches(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> int:
    """Conta o número de trocas de identidade (ID switches / IDSW).

    Uma troca ocorre quando um objeto de GT associado ao pred_id A passa a ser
    associado ao pred_id B (A != B).
    """
    # Mapear por frame
    frames_gt: Dict[int, Dict[int, np.ndarray]] = {}
    for g_id, frames in gt_tracks.items():
        for f, bbox in frames.items():
            if f not in frames_gt:
                frames_gt[f] = {}
            frames_gt[f][g_id] = bbox

    frames_pred: Dict[int, Dict[int, np.ndarray]] = {}
    for p_id, frames in pred_tracks.items():
        for f, bbox in frames.items():
            if f not in frames_pred:
                frames_pred[f] = {}
            frames_pred[f][p_id] = bbox

    all_frames = sorted(set(frames_gt.keys()) | set(frames_pred.keys()))

    last_associated_pred: Dict[int, int] = {}
    id_switches = 0

    for f in all_frames:
        g_in_frame = frames_gt.get(f, {})
        p_in_frame = frames_pred.get(f, {})

        if not g_in_frame or not p_in_frame:
            continue

        g_keys = list(g_in_frame.keys())
        p_keys = list(p_in_frame.keys())

        g_boxes = np.array([g_in_frame[k] for k in g_keys], dtype=np.float32)
        p_boxes = np.array([p_in_frame[k] for k in p_keys], dtype=np.float32)

        iou_mat = compute_iou_matrix(g_boxes, p_boxes)
        cost_mat = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_threshold:
                g_id = g_keys[r]
                p_id = p_keys[c]

                if g_id in last_associated_pred:
                    if last_associated_pred[g_id] != p_id:
                        id_switches += 1
                last_associated_pred[g_id] = p_id

    return id_switches


def count_fragmentations(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> int:
    """Conta fragmentações de trajetórias.

    Uma fragmentação ocorre quando um track de GT passa de um estado 'sendo rastreado'
    para 'não rastreado'.
    """
    frames_gt: Dict[int, Dict[int, np.ndarray]] = {}
    for g_id, frames in gt_tracks.items():
        for f, bbox in frames.items():
            if f not in frames_gt:
                frames_gt[f] = {}
            frames_gt[f][g_id] = bbox

    frames_pred: Dict[int, Dict[int, np.ndarray]] = {}
    for p_id, frames in pred_tracks.items():
        for f, bbox in frames.items():
            if f not in frames_pred:
                frames_pred[f] = {}
            frames_pred[f][p_id] = bbox

    # Identificar se cada GT está associado em cada frame
    all_frames = sorted(frames_gt.keys())
    is_tracked_prev: Dict[int, bool] = {}
    fragmentations = 0

    for f in all_frames:
        g_in_frame = frames_gt[f]
        p_in_frame = frames_pred.get(f, {})

        matched_gt = set()
        if p_in_frame:
            g_keys = list(g_in_frame.keys())
            p_keys = list(p_in_frame.keys())
            g_boxes = np.array([g_in_frame[k] for k in g_keys], dtype=np.float32)
            p_boxes = np.array([p_in_frame[k] for k in p_keys], dtype=np.float32)

            iou_mat = compute_iou_matrix(g_boxes, p_boxes)
            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= iou_threshold:
                    matched_gt.add(g_keys[r])

        for g_id in g_in_frame.keys():
            currently_tracked = (g_id in matched_gt)
            if g_id in is_tracked_prev:
                # Transição de tracked -> untracked é uma fragmentação
                if is_tracked_prev[g_id] and not currently_tracked:
                    fragmentations += 1
            is_tracked_prev[g_id] = currently_tracked

    return fragmentations


def count_unique_id_error(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
) -> int:
    """Erro absoluto na contagem de identidades únicas: |N_pred - N_gt|."""
    return abs(len(pred_tracks) - len(gt_tracks))


def evaluate_sequence(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict:
    """Executa a suíte de métricas oficial sobre uma sequência completa."""
    idf1 = compute_idf1(gt_tracks, pred_tracks, iou_threshold)
    ids = count_id_switches(gt_tracks, pred_tracks, iou_threshold)
    frag = count_fragmentations(gt_tracks, pred_tracks, iou_threshold)
    uid_err = count_unique_id_error(gt_tracks, pred_tracks)

    return {
        'idf1': idf1,
        'id_switches': ids,
        'fragmentations': frag,
        'unique_id_error': uid_err,
    }
