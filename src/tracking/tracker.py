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


@dataclass
class TrackState:
    """Estado de uma trajetória ativa no tracker."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    age: int = 0  # total de frames desde nascimento
    hits: int = 0  # total de detecções associadas
    time_since_update: int = 0  # frames consecutivos sem associação
    rnn_hidden: Optional[torch.Tensor] = None  # estado oculto da RNN (para TemporalTracker)


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
        raise NotImplementedError("Isaias: implementar lógica de ciclo de vida e associação do BaselineTracker.")


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
    ):
        super().__init__(iou_threshold, max_age, min_hits, matching)
        self.motion_model = motion_model
        self.device = device

    def update(self, detections: List[Dict]) -> List[TrackState]:
        """Executa o ciclo temporal de predição -> matching -> atualização de hidden state."""
        raise NotImplementedError("Isaias: implementar matching temporal com predições da MotionRNN.")
