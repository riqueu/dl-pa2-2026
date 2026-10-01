"""Módulo de provedores de detecção para o MOT17.

Requisitos do Edital (Parte 1):
- Suporte a duas fontes de detecção:
  1. Detecções públicas pré-computadas do MOT17 (SDP/DPM/FRCNN).
  2. Detector pré-treinado do Torchvision (ex.: Faster R-CNN COCO pessoa).
- As predições do Torchvision devem passar pelo NMS autoral (`src.nms.nms`).

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
"""

from typing import Dict, List, Optional
import numpy as np
import torch


class MOT17DetectionLoader:
    """Carregador de detecções pré-computadas da pasta det/det.txt de uma sequência MOT17."""

    def __init__(self, seq_path: str, conf_threshold: float = 0.0):
        self.seq_path = seq_path
        self.conf_threshold = conf_threshold
        # Dica: use src.data.mot17.load_detections para carregar do disco
        raise NotImplementedError("Isaias: implementar inicialização do carregador MOT17.")

    def get_detections(self, frame_id: int) -> List[Dict]:
        """Retorna lista de detecções do frame: [{'bbox': np.array([x1, y1, x2, y2]), 'confidence': float}]."""
        raise NotImplementedError("Isaias: implementar busca de detecções por frame.")


class TorchvisionDetector:
    """Wrapper para detector Faster R-CNN pré-treinado da torchvision, com NMS autoral."""

    def __init__(
        self,
        model_name: str = 'fasterrcnn_resnet50_fpn_v2',
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        raise NotImplementedError("Isaias: carregar modelo Faster R-CNN pré-treinado e mover para device.")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Executa inferência na imagem, filtra classe 1 (person), aplica NMS autoral.

        Args:
            frame: Imagem RGB ou BGR (H, W, 3) em uint8.

        Returns:
            Lista de detecções com bboxes em [x1, y1, x2, y2] e confidence score.
        """
        raise NotImplementedError("Isaias: rodar inferência do Faster R-CNN e filtrar com NMS autoral.")
