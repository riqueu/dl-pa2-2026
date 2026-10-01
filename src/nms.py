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
    raise NotImplementedError("Isaias: implementar algoritmo NMS com NumPy.")
