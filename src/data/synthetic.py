"""Módulo de geração de dados sintéticos e simulação de detector para o PA2 (Parte 0).

Responsável: Membro 1 (Henrique)
Branch: feature/data-metrics-association
Teste unitário: pytest tests/test_synthetic.py
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2


def generate_synthetic_video(
    n_frames: int = 45,
    n_objects: int = 10,
    img_size: int = 128,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[int, Dict[int, np.ndarray]], Dict[int, List[Dict]]]:
    """Gera um vídeo sintético de elipses móveis com oclusões reais via z-buffer.

    Requisitos do Edital (Parte 0):
        - Canvas 128x128, entre 30 e 60 frames (default: 45).
        - Entre 5 e 15 elipses com tamanhos variados, ruído e contraste.
        - Suporte a oclusão real (ordenação de profundidade / depth ordering):
          elipses com menor profundidade devem ser sobrepostas por elipses mais próximas.
        - Rebate nas bordas do canvas para manter os objetos em cena.

    Args:
        n_frames: Quantidade de quadros no vídeo.
        n_objects: Quantidade de objetos móveis.
        img_size: Resolução quadrada (128x128).
        noise_std: Desvio-padrão do ruído gaussiano adicionado.
        seed: Semente pseudoaleatória para reprodutibilidade.

    Returns:
        frames: Array uint8 (n_frames, img_size, img_size, 3) no formato BGR ou RGB.
        gt_tracks: Dict indexado por track_id: {frame_id: bbox_xyxy em np.ndarray}.
        gt_per_frame: Dict indexado por frame_id: [{'track_id': int, 'bbox': np.ndarray, 'visibility': float}].
    """
    raise NotImplementedError("Henrique: implementar gerador sintético procedural com oclusões via z-buffer.")


def degrade_detections(
    gt_per_frame: Dict[int, List[Dict]],
    drop_rate: float = 0.1,
    noise_std: float = 3.0,
    fp_rate: float = 0.05,
    img_size: int = 128,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """Simulador de detector imperfeito sobre as anotações perfeitas do GT.

    Requisitos do Edital (Parte 0):
        - Recebe as bounding boxes perfeitas e degrada propositalmente:
          1. Dropa p% das detecções (falsos negativos).
          2. Adiciona ruído gaussiano nas coordenadas [x1, y1, x2, y2].
          3. Injeta falsos positivos espalhados pelo canvas.

    Args:
        gt_per_frame: Dicionário retornado pelo gerador sintético.
        drop_rate: Fração de detecções a remover aleatoriamente.
        noise_std: Desvio padrão do ruído adicionado às coordenadas.
        fp_rate: Taxa de injeção de falsos positivos por frame.
        img_size: Tamanho do frame para limitar coordenadas válidas.
        seed: Semente pseudoaleatória.

    Returns:
        Dicionário {frame_id: [{'bbox': np.ndarray, 'confidence': float}, ...]}.
    """
    raise NotImplementedError("Henrique: implementar simulador de degradação de detecções.")


def create_metric_edge_cases() -> List[Tuple[Dict, Dict, str]]:
    """Cria os 3 cenários de teste manuais exigidos pelo edital para validação das métricas.

    Edge Cases Obrigatórios:
        1. Predição perfeita: pred_tracks == gt_tracks -> IDF1 ≈ 1.0, ID switches = 0, frag = 0.
        2. IDs trocados no meio da sequência -> ID switches = 2, IDF1 cai.
        3. Trajetória dividida (split tracks com gap) -> fragmentations >= 1.

    Returns:
        Lista de tuplas (gt_tracks, pred_tracks, descricao).
    """
    raise NotImplementedError("Henrique: montar os 3 edge cases manuais do edital.")
