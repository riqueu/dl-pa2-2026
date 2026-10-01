"""Módulo da rede neural recorrente para predição de movimento (Trilha A).

Implementa:
- MotionRNN: módulo PyTorch parametrizável por cell_type ('rnn', 'lstm', 'gru').
- extract_training_sequences: conversão de GT do MOT17 em tensores de trajetória normalizados.
- train_motion_model: loop de treinamento com Truncated BPTT (TBPTT).

Responsável: Membro 2 (Isaias)
Branch: feature/detection-tracking-model
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn

HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


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
        cells = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if self.cell_type not in cells or input_dim != 4:
            raise ValueError("Use input_dim=4 e cell_type rnn, lstm ou gru.")
        self.recurrent = cells[self.cell_type](
            input_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0)
        self.projection = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[HiddenState] = None,
    ) -> Tuple[torch.Tensor, HiddenState]:
        """Forward pass sobre uma sequência de bounding boxes.

        Args:
            obs_seq: Tensor (seq_len, batch, 4) ou (seq_len, 4).
            hidden: Estado oculto anterior.

        Returns:
            pred_seq: Bboxes preditos para cada instante.
            new_hidden: Novo estado oculto.
        """
        if obs_seq.ndim not in (2, 3) or obs_seq.shape[-1] != 4:
            raise ValueError("Esperado tensor (T, 4) ou (T, B, 4).")
        unbatched = obs_seq.ndim == 2
        if unbatched:
            obs_seq = obs_seq.unsqueeze(1)
        output, new_hidden = self.recurrent(obs_seq, hidden)
        prediction = self.projection(output)
        return (prediction.squeeze(1) if unbatched else prediction), new_hidden

    def predict_single(self, hidden: HiddenState) -> Tuple[np.ndarray, HiddenState]:
        """Prediz 1 passo à frente a partir do estado oculto atual (usado no tracker online)."""
        # O estado já consumiu a última observação; projetá-lo não avança a RNN.
        state = hidden[0] if isinstance(hidden, tuple) else hidden
        if state.shape[1] != 1:
            raise ValueError("predict_single requer batch_size=1.")
        with torch.no_grad():
            prediction = self.projection(state[-1, 0])
        return prediction.detach().cpu().numpy(), hidden

    def init_hidden(self, batch_size: int = 1) -> HiddenState:
        """Inicializa tensor de zeros para o estado oculto."""
        param = next(self.parameters())
        h = param.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h, h.clone()) if self.cell_type == "lstm" else h


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
    if img_width <= 0 or img_height <= 0 or min_length < 2:
        raise ValueError("Dimensões positivas e min_length >= 2 são necessários.")
    sequences = []

    def append_segment(segment):
        if len(segment) >= min_length:
            boxes = np.asarray(segment, dtype=np.float32)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = boxes[:, 2:] - boxes[:, :2]
            normalized = np.concatenate((centers, sizes), axis=1)
            normalized /= np.array([img_width, img_height, img_width, img_height])
            sequences.append(torch.from_numpy(normalized))
    for frames in gt_tracks.values():
        segment, previous = [], None
        for frame_id, bbox in sorted(frames.items()):
            bbox = np.asarray(bbox, dtype=np.float32)
            valid = (bbox.shape == (4,) and np.isfinite(bbox).all()
                     and np.all(bbox[2:] > bbox[:2]))
            if (previous is not None and frame_id != previous + 1) or not valid:
                append_segment(segment)
                segment = []
            if valid:
                segment.append(bbox)
            previous = frame_id
        append_segment(segment)
    return sequences


def train_motion_model(
    model: nn.Module,
    sequences: List[torch.Tensor],
    epochs: int = 50,
    lr: float = 1e-3,
    tbptt_len: int = 16,
    max_grad_norm: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 1,
) -> Dict:
    """Loop de treinamento com Truncated Backpropagation Through Time (TBPTT).

    Loss: SmoothL1Loss entre o bbox predito no instante t e o observado em t+1.
    A cada janela de `tbptt_len` frames, os gradientes são truncados (`detach()`).

    Returns:
        Histórico de treinamento com chave 'loss'.
    """
    if epochs < 1 or tbptt_len < 1 or batch_size < 1 or max_grad_norm <= 0:
        raise ValueError("epochs, tbptt_len, batch_size e max_grad_norm devem ser positivos.")
    usable = [seq for seq in sequences if len(seq) >= 2]
    if not usable:
        raise ValueError("Nenhuma trajetória com pelo menos dois frames.")
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": []}
    # Agrupar por comprimento evita padding e mantém estados independentes.
    groups = {}
    for seq in usable:
        groups.setdefault(len(seq), []).append(seq)
    for _ in range(epochs):
        total, elements = 0.0, 0
        for group in groups.values():
            order = torch.randperm(len(group)).tolist()
            for start in range(0, len(order), batch_size):
                batch = torch.stack([group[i] for i in order[start:start + batch_size]], dim=1).to(device)
                hidden = None
                for offset in range(0, len(batch) - 1, tbptt_len):
                    end = min(offset + tbptt_len, len(batch) - 1)
                    optimizer.zero_grad()
                    prediction, hidden = model(batch[offset:end], hidden)
                    target = batch[offset + 1:end + 1]
                    loss = nn.functional.smooth_l1_loss(prediction, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
                    total += loss.item() * target.numel()
                    elements += target.numel()
        history["loss"].append(total / elements)
    return history
