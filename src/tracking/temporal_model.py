"""Módulo da rede neural recorrente para predição de movimento (Trilha A).

Implementa:
- MotionRNN: módulo PyTorch parametrizável por cell_type ('rnn', 'lstm', 'gru').
- extract_training_sequences: conversão de GT do MOT17 em tensores de trajetória normalizados.
- train_motion_model: loop de treinamento com Truncated BPTT (TBPTT).

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


class MotionRNN(nn.Module):
    """Modelo recorrente para previsão de bounding box futuro (Trilha A).

    Entrada: sequência de bboxes normalizados [cx, cy, w, h] em [0, 1].
    Saída: predição do próximo bbox [cx, cy, w, h].
    Suporta células 'rnn' (SimpleRNN), 'lstm' e 'gru' para atender ao Eixo 1 de ablações.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        cell_type: str = 'gru',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        raise NotImplementedError("Isaias: inicializar célula recorrente (RNN/LSTM/GRU) e camada linear de projeção.")

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass sobre uma sequência de bounding boxes.

        Args:
            obs_seq: Tensor (seq_len, batch, 4) ou (seq_len, 4).
            hidden: Estado oculto anterior.

        Returns:
            pred_seq: Bboxes preditos para cada instante.
            new_hidden: Novo estado oculto.
        """
        raise NotImplementedError("Isaias: implementar forward pass da MotionRNN.")

    def predict_single(self, hidden: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Prediz 1 passo à frente a partir do estado oculto atual (usado no tracker online)."""
        raise NotImplementedError("Isaias: implementar predição de passo único.")

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Inicializa tensor de zeros para o estado oculto."""
        raise NotImplementedError("Isaias: implementar inicialização de hidden state.")


def extract_training_sequences(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],
    min_length: int = 10,
    img_width: float = 1920.0,
    img_height: float = 1080.0,
) -> List[torch.Tensor]:
    """Extrai sequências de trajetórias dos tracks de GT para treino da RNN.

    Converte bboxes [x1, y1, x2, y2] para coordenadas normalizadas [cx, cy, w, h].

    Args:
        gt_tracks: Dict {track_id: {frame_id: bbox}} vindo de `src.data.mot17.load_ground_truth`.
        min_length: Comprimento mínimo de trajetória para aproveitar no treino.
        img_width: Largura da imagem da sequência (para normalização em [0, 1]).
        img_height: Altura da imagem da sequência (para normalização em [0, 1]).

    Returns:
        Lista de tensores (seq_len, 4) prontos para treinamento.
    """
    raise NotImplementedError("Isaias: extrair e normalizar trajetórias temporais do GT.")


def train_motion_model(
    model: nn.Module,
    sequences: List[torch.Tensor],
    epochs: int = 50,
    lr: float = 1e-3,
    tbptt_len: int = 16,
    max_grad_norm: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict:
    """Loop de treinamento com Truncated Backpropagation Through Time (TBPTT).

    Loss: SmoothL1Loss entre o bbox predito no instante t e o observado em t+1.
    A cada janela de `tbptt_len` frames, os gradientes são truncados (`detach()`).

    Returns:
        Histórico de treinamento com chave 'loss'.
    """
    raise NotImplementedError("Isaias: implementar loop de treinamento TBPTT da MotionRNN.")
