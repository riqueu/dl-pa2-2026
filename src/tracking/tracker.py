"""Módulo de rastreadores multi-objeto (Trackers).

Contém:
- TrackState: representação do estado interno de cada objeto rastreado.
- BaselineTracker: rastreador frame-a-frame ingênuo (Parte 1).
- TemporalTracker: rastreador guiado por memória temporal RNN (Parte 2).

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from .association import compute_iou_matrix, greedy_matching, hungarian_matching
from .temporal_model import HiddenState


@dataclass
class TrackState:
    """Estado de uma trajetória ativa no tracker."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    age: int = 0  # total de frames desde nascimento
    hits: int = 0  # total de detecções associadas
    time_since_update: int = 0  # frames consecutivos sem associação
    rnn_hidden: Optional[HiddenState] = None  # estado oculto da RNN (para TemporalTracker)


class BaselineTracker:
    """Rastreador frame-a-frame da Parte 1 baseado puramente em IoU espacial.

    Regras de associação e ciclo de vida:
        - Associação via IoU (Hungarian ou Greedy) com detecções do frame anterior.
        - Birth: detecções não associadas iniciam novo track.
        - Hits: tracks só são reportados publicamente após min_hits frames associados.
        - Death: tracks sem associação por mais de max_age frames são eliminados.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        matching: str = 'hungarian',
    ):
        if matching not in ("hungarian", "greedy"):
            raise ValueError("matching deve ser hungarian ou greedy.")
        if not 0 <= iou_threshold <= 1 or max_age < 0 or min_hits < 1:
            raise ValueError("Parâmetros de ciclo de vida inválidos.")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching = matching
        self.tracks: List[TrackState] = []
        self.next_id: int = 1

    def reset(self) -> None:
        """Reinicia o estado para uma nova sequência."""
        self.tracks = []
        self.next_id = 1

    def update(self, detections: List[Dict]) -> List[TrackState]:
        """Processa um frame de detecções e atualiza o estado dos tracks.

        Args:
            detections: Lista de dicts com chave 'bbox' em [x1, y1, x2, y2].

        Returns:
            Lista de tracks confirmados ativos no frame atual.
        """
        self._associate(detections)
        # Baseline guarda tracks perdidos, mas só reporta observações atuais.
        return [track for track in self.tracks
                if track.hits >= self.min_hits and track.time_since_update == 0]

    def _associate(self, detections):
        boxes = np.asarray([d["bbox"] for d in detections], dtype=np.float32).reshape(-1, 4)
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Detecções devem ter caixas finitas com área positiva.")
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
        track_boxes = np.asarray([t.bbox for t in self.tracks]).reshape(-1, 4)
        matcher = hungarian_matching if self.matching == "hungarian" else greedy_matching
        matches, _, unmatched = matcher(compute_iou_matrix(track_boxes, boxes), self.iou_threshold)
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            track.bbox = boxes[det_idx].copy()
            track.hits += 1
            track.time_since_update = 0
        for det_idx in unmatched:
            self.tracks.append(TrackState(self.next_id, boxes[det_idx].copy(), age=1, hits=1))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]


class TemporalTracker(BaselineTracker):
    """Rastreador da Parte 2 (Trilha A) guiado por predição temporal via MotionRNN.

    Diferencial sobre o baseline:
        1. Antes da associação, a MotionRNN prevê o próximo bbox estimado para cada track ativo.
        2. A matriz de IoU é calculada entre bboxes PREDITOS e DETECÇÕES observadas.
        3. Para tracks com match: alimenta a observação real na RNN e atualiza seu hidden state.
        4. Para tracks sem match (oclusão): mantém a trajetória 'coasting' com o bbox predito
           e alimenta a própria predição na RNN, mantendo o track vivo por até max_age frames.
    """

    def __init__(
        self,
        motion_model: nn.Module,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        matching: str = 'hungarian',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        img_width: float = 1920.0,
        img_height: float = 1080.0,
        max_shift: float = 30.0,
        max_coast: Optional[int] = None,
    ):
        super().__init__(iou_threshold, max_age, min_hits, matching)
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Dimensões da imagem devem ser positivas.")
        self.motion_model = motion_model.to(device).eval()
        self.device = device
        self.scale = np.array([img_width, img_height], dtype=np.float32)
        self.max_shift = max_shift
        self.max_coast = max_age if max_coast is None and max_age <= 2 else (0 if max_coast is None else max_coast)

    def update(self, detections: List[Dict]) -> List[TrackState]:
        """Executa o ciclo temporal de predição -> matching -> atualização de hidden state."""
        with torch.no_grad():
            for track in self.tracks:
                if track.rnn_hidden is not None:
                    prediction, _ = self.motion_model.predict_single(track.rnn_hidden)
                    prediction = np.asarray(prediction, dtype=np.float32)
                    if prediction.shape == (4,) and np.isfinite(prediction).all():
                        center = prediction[:2] * self.scale
                        size = np.maximum(prediction[2:] * self.scale, 1.0)
                        pred_box = np.concatenate((center - size / 2, center + size / 2))
                        old_center = (track.bbox[:2] + track.bbox[2:]) / 2
                        # Ancoragem de plausibilidade física: só aceita se deslocamento for viável
                        if np.linalg.norm(center - old_center) <= self.max_shift:
                            track.bbox = pred_box
            self._associate(detections)
            for track in self.tracks:
                if track.time_since_update == 0:
                    center = (track.bbox[:2] + track.bbox[2:]) / 2 / self.scale
                    size = (track.bbox[2:] - track.bbox[:2]) / self.scale
                    observation = torch.as_tensor(np.concatenate((center, size)),
                                                  dtype=torch.float32, device=self.device).reshape(1, 1, 4)
                    _, track.rnn_hidden = self.motion_model(observation, track.rnn_hidden)
        # Retorna apenas tracks ativos (sem caixas fantasmas de objetos perdidos)
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update <= self.max_coast]

