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
    raise NotImplementedError("Henrique: implementar cálculo de IoU entre dois bboxes.")


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Calcula a matriz de IoU entre duas listas de bboxes de forma vetorizada.

    Args:
        bboxes1: Array (N, 4) no formato [x1, y1, x2, y2].
        bboxes2: Array (M, 4) no formato [x1, y1, x2, y2].

    Returns:
        Array numpy (N, M) com o IoU par a par.
    """
    raise NotImplementedError("Henrique: implementar matriz de IoU vetorizada.")


def compute_idf1(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> float:
    """Calcula a métrica IDF1 (ID F1-score) global da sequência.

    Definição:
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Passos recomendados:
        1. Construir matriz de custo/pesos entre cada track de GT e cada track Predito:
           peso(gt_id, pred_id) = número de frames em que IoU(gt, pred) >= iou_threshold.
        2. Resolver o casamento bipartido global ótimo usando `linear_sum_assignment`
           da scipy para maximizar o número total de identificações corretas.
        3. Para os pares casados, somar as sobreposições válidas como IDTP.
        4. IDFP = total_pred_detections - IDTP
        5. IDFN = total_gt_detections - IDTP
        6. Retornar 2 * IDTP / (2 * IDTP + IDFP + IDFN) (ou 0.0 se denominador for 0).

    Args:
        gt_tracks: {gt_id: {frame_id: bbox_xyxy}}
        pred_tracks: {pred_id: {frame_id: bbox_xyxy}}
        iou_threshold: Limiar mínimo de IoU espacial para considerar casamento válido.

    Returns:
        float: Pontuação IDF1 em [0.0, 1.0].
    """
    raise NotImplementedError("Henrique: implementar cálculo global de IDF1 via casamento bipartido.")


def count_id_switches(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> int:
    """Conta o número de trocas de identidade (ID switches / IDSW).

    Uma troca de identidade ocorre quando um objeto de GT que vinha sendo associado
    ao pred_id X passa a ser associado a um pred_id Y diferente (em que ambos têm
    IoU >= iou_threshold com o GT naquele frame).

    Args:
        gt_tracks: {gt_id: {frame_id: bbox_xyxy}}
        pred_tracks: {pred_id: {frame_id: bbox_xyxy}}
        iou_threshold: Limiar mínimo de IoU.

    Returns:
        int: Contagem total de ID switches na sequência.
    """
    raise NotImplementedError("Henrique: implementar contagem de trocas de identidade ao longo do tempo.")


def count_fragmentations(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    pred_tracks: Dict[int, Dict[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> int:
    """Conta fragmentações de trajetórias.

    Uma fragmentação ocorre quando um track de GT passa de um estado 'sendo rastreado'
    (com detecção predita associada) para 'não rastreado' (miss / oclusão / perda).

    Args:
        gt_tracks: {gt_id: {frame_id: bbox_xyxy}}
        pred_tracks: {pred_id: {frame_id: bbox_xyxy}}
        iou_threshold: Limiar de IoU.

    Returns:
        int: Contagem total de fragmentações.
    """
    raise NotImplementedError("Henrique: implementar contagem de fragmentações.")


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
