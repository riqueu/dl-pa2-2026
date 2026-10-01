"""Módulo de Non-Maximum Suppression (NMS) autoral.

Regras do PA2:
- Proibido o uso de torchvision.ops.nms.
- Deve ser implementado do zero com NumPy.

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
Teste unitário: pytest tests/test_nms.py
"""

import numpy as np


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Aplica Non-Maximum Suppression (NMS) sobre caixas delimitadoras.

    Algoritmo:
        1. Ordenar as caixas pelos scores de forma decrescente.
        2. Selecionar a caixa de maior score e guardá-la na lista de mantidos.
        3. Calcular a sobreposição (IoU) desta caixa com todas as restantes.
        4. Suprimir (descartar) todas as caixas cujo IoU for superior a iou_threshold.
        5. Repetir os passos 2 a 4 até não sobrarem caixas na fila.

    Args:
        boxes: Array numpy (N, 4) no formato [x1, y1, x2, y2].
        scores: Array numpy (N,) com a confiança de cada caixa.
        iou_threshold: Limiar de IoU a partir do qual caixas sobrepostas são suprimidas.

    Returns:
        Array numpy 1D com os índices das caixas selecionadas/mantidas.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.shape != (len(scores), 4) or scores.ndim != 1:
        raise ValueError("Esperado boxes (N, 4) e scores (N,).")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold deve estar em [0, 1].")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Caixas e scores devem ser finitos.")
    order = np.argsort(-scores, kind="stable")
    size = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    areas = size[:, 0] * size[:, 1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        intersection_size = np.maximum(
            np.minimum(boxes[current, 2:], boxes[rest, 2:])
            - np.maximum(boxes[current, :2], boxes[rest, :2]), 0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)
