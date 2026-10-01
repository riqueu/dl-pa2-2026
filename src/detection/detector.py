"""Módulo de provedores de detecção para o MOT17.

Requisitos do Edital (Parte 1):
- Suporte a duas fontes de detecção:
  1. Detecções públicas pré-computadas do MOT17 (SDP/DPM/FRCNN).
  2. Detector pré-treinado do Torchvision (ex.: Faster R-CNN COCO pessoa).
- As predições do Torchvision devem passar pelo NMS autoral (`src.nms.nms`).

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
"""

from typing import Dict, List
import numpy as np
import torch
from types import FunctionType, MethodType, SimpleNamespace

from src.data.mot17 import load_detections
from src.nms import nms


def _batched_nms(boxes, scores, labels, threshold):
    """NMS autoral independente por classe/nível, ordenado por confiança."""
    kept = []
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        local = nms(boxes[indices].detach().cpu().numpy(),
                    scores[indices].detach().cpu().numpy(), threshold)
        kept.append(indices[torch.as_tensor(local, device=boxes.device)])
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    kept = torch.cat(kept)
    return kept[torch.argsort(scores[kept], descending=True, stable=True)]


def _install_custom_nms(model):
    """Substitui NMS de RPN/RoI somente nesta instância, sem patch global.

    Preserva clipping, seleção de propostas e decodificação do torchvision.
    As funções recebem uma cópia de seus globals com box_ops restrito.
    """
    from torchvision.ops import boxes as box_ops
    custom_ops = SimpleNamespace(
        clip_boxes_to_image=box_ops.clip_boxes_to_image,
        remove_small_boxes=box_ops.remove_small_boxes,
        batched_nms=_batched_nms)
    for component, name in ((model.rpn, "filter_proposals"),
                            (model.roi_heads, "postprocess_detections")):
        method = getattr(component, name).__func__
        namespace = dict(method.__globals__, box_ops=custom_ops)
        replacement = FunctionType(method.__code__, namespace, method.__name__,
                                   method.__defaults__, method.__closure__)
        setattr(component, name, MethodType(replacement, component))


class MOT17DetectionLoader:
    """Carregador de detecções pré-computadas da pasta det/det.txt de uma sequência MOT17."""

    def __init__(self, seq_path: str, conf_threshold: float = 0.0):
        self.seq_path = seq_path
        self.conf_threshold = conf_threshold
        # Dica: use src.data.mot17.load_detections para carregar do disco
        self.detections = load_detections(seq_path)

    def get_detections(self, frame_id: int) -> List[Dict]:
        """Retorna lista de detecções do frame: [{'bbox': np.array([x1, y1, x2, y2]), 'confidence': float}]."""
        return [{"bbox": d["bbox"].copy(), "confidence": d["confidence"]}
                for d in self.detections.get(frame_id, [])
                if d["confidence"] >= self.conf_threshold]


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
        from torchvision.models import get_model, get_model_weights
        if model_name not in ("fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"):
            raise ValueError("Modelo Faster R-CNN não suportado.")
        weights = get_model_weights(model_name).DEFAULT
        self.model = get_model(model_name, weights=weights)
        _install_custom_nms(self.model)
        self.model.to(device).eval()

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Executa inferência na imagem, filtra classe 1 (person), aplica NMS autoral.

        Args:
            frame: Imagem RGB (H, W, 3) em uint8.

        Returns:
            Lista de detecções com bboxes em [x1, y1, x2, y2] e confidence score.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Esperada imagem RGB uint8 (H, W, 3).")
        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(self.device)
        image = image.float() / 255.0
        with torch.inference_mode():
            output = self.model([image])[0]
        mask = (output["labels"] == 1) & (output["scores"] >= self.conf_threshold)
        boxes = output["boxes"][mask].cpu().numpy()
        scores = output["scores"][mask].cpu().numpy()
        kept = nms(boxes, scores, self.nms_threshold)
        return [{"bbox": boxes[i].copy(), "confidence": float(scores[i])} for i in kept]
