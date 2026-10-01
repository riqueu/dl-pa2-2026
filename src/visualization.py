"""Módulo de visualização para o tracker MOT.

Implementa:
1. Renderização de bounding boxes coloridos por identidade em frames
2. Geração de strips visuais de falha (sequência de frames lado a lado)
3. Plots de métricas ao longo do tempo
4. Geração de vídeo com identidades rastreadas
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_distinct_colors(n_colors: int, seed: int = 42) -> np.ndarray:
    """Gera N cores RGB distintas para colorir identidades de tracks.

    Args:
        n_colors: Número de cores necessárias.
        seed: Semente para reprodutibilidade.

    Returns:
        Array (n_colors, 3) com valores RGB em [0, 255].
    """
    if n_colors <= 0:
        return np.zeros((1, 3), dtype=np.uint8)

    rng = np.random.RandomState(seed)
    hues = np.linspace(0.0, 1.0, n_colors, endpoint=False)
    rng.shuffle(hues)

    colors = np.zeros((n_colors, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        # HSV -> RGB com saturação e valor altos
        s = rng.uniform(0.7, 1.0)
        v = rng.uniform(0.8, 1.0)
        c = v * s
        x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
        m = v - c
        if h < 1 / 6:
            rgb = (c + m, x + m, m)
        elif h < 2 / 6:
            rgb = (x + m, c + m, m)
        elif h < 3 / 6:
            rgb = (m, c + m, x + m)
        elif h < 4 / 6:
            rgb = (m, x + m, c + m)
        elif h < 5 / 6:
            rgb = (x + m, m, c + m)
        else:
            rgb = (c + m, m, x + m)
        colors[i] = [int(c * 255) for c in rgb]

    return colors


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: List[Dict],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Desenha bounding boxes coloridos por identidade em um frame.

    Args:
        frame: Imagem BGR (H, W, 3) uint8.
        tracks: Lista de dicts com 'track_id' e 'bbox' [x1, y1, x2, y2].
        color_map: Mapa {track_id: (B, G, R)}. Se None, gera automaticamente.
        thickness: Espessura das linhas dos bboxes.
        font_scale: Escala do texto do ID.

    Returns:
        Frame anotado (cópia).
    """
    vis = frame.copy()

    if color_map is None:
        all_ids = sorted(set(t['track_id'] for t in tracks))
        colors = generate_distinct_colors(len(all_ids))
        color_map = {tid: tuple(int(c) for c in colors[i]) for i, tid in enumerate(all_ids)}

    for t in tracks:
        tid = t['track_id']
        bbox = t['bbox'].astype(int)
        color = color_map.get(tid, (255, 255, 255))

        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        label = f"ID:{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (bbox[0], bbox[1] - th - 6), (bbox[0] + tw + 4, bbox[1]), color, -1)
        cv2.putText(
            vis, label, (bbox[0] + 2, bbox[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def create_failure_strip(
    frames: List[np.ndarray],
    tracks_per_frame: List[List[Dict]],
    gt_per_frame: Optional[List[List[Dict]]] = None,
    max_frames: int = 8,
    resize_height: int = 200,
) -> np.ndarray:
    """Cria uma strip horizontal de frames mostrando um caso de falha.

    Args:
        frames: Lista de frames BGR.
        tracks_per_frame: Lista de tracks por frame.
        gt_per_frame: Lista de GT por frame (para comparação, opcional).
        max_frames: Número máximo de frames a mostrar.
        resize_height: Altura para redimensionar cada frame.

    Returns:
        Strip horizontal (H, W_total, 3) uint8.
    """
    step = max(1, len(frames) // max_frames)
    selected_indices = list(range(0, len(frames), step))[:max_frames]

    annotated = []
    for idx in selected_indices:
        vis = draw_tracks_on_frame(frames[idx], tracks_per_frame[idx])
        h, w = vis.shape[:2]
        scale = resize_height / h
        vis_resized = cv2.resize(vis, (int(w * scale), resize_height))
        annotated.append(vis_resized)

    return np.concatenate(annotated, axis=1)


def save_tracking_video(
    frames: List[np.ndarray],
    tracks_per_frame: List[List[Dict]],
    output_path: str,
    fps: int = 15,
) -> None:
    """Salva um vídeo com identidades rastreadas e coloridas.

    Args:
        frames: Lista de frames BGR.
        tracks_per_frame: Lista de tracks por frame.
        output_path: Caminho de saída (.mp4 ou .avi).
        fps: Frames por segundo do vídeo.
    """
    # Coletar todos os IDs para colorização consistente
    all_ids = set()
    for tf in tracks_per_frame:
        for t in tf:
            all_ids.add(t['track_id'])

    colors = generate_distinct_colors(len(all_ids))
    color_map = {tid: tuple(int(c) for c in colors[i]) for i, tid in enumerate(sorted(all_ids))}

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame, tracks in zip(frames, tracks_per_frame):
        vis = draw_tracks_on_frame(frame, tracks, color_map=color_map)
        writer.write(vis)

    writer.release()


def plot_metrics_over_sequences(
    results: Dict[str, Dict[str, float]],
    metric_name: str = 'idf1',
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plota uma métrica por sequência, ordenada por dificuldade.

    Args:
        results: {seq_name: {metric_name: value, 'n_gt_tracks': int, ...}}
        metric_name: Nome da métrica a plotar.
        output_path: Caminho para salvar a figura.
        title: Título do gráfico.
    """
    seq_names = sorted(results.keys(), key=lambda s: results[s].get('n_gt_tracks', 0))
    values = [results[s][metric_name] for s in seq_names]
    densities = [results[s].get('n_gt_tracks', 0) for s in seq_names]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_bar = '#2196F3'
    color_line = '#FF5722'

    ax1.bar(range(len(seq_names)), values, color=color_bar, alpha=0.7, label=metric_name.upper())
    ax1.set_xlabel('Sequência (ordenada por nº de tracks GT)')
    ax1.set_ylabel(metric_name.upper(), color=color_bar)
    ax1.set_xticks(range(len(seq_names)))
    ax1.set_xticklabels([s.replace('MOT17-', '') for s in seq_names], rotation=45)
    ax1.tick_params(axis='y', labelcolor=color_bar)

    ax2 = ax1.twinx()
    ax2.plot(range(len(seq_names)), densities, 'o-', color=color_line, label='Nº GT Tracks')
    ax2.set_ylabel('Nº GT Tracks', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)

    plt.title(title or f'{metric_name.upper()} por Sequência')
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)


def plot_gradient_horizon(
    grad_norms: Dict[int, float],
    output_path: Optional[str] = None,
    title: str = 'Horizonte de Memória — Norma do Gradiente vs. k',
) -> None:
    """Plota a norma do gradiente ∂L_t/∂h_{t-k} em função de k.

    Args:
        grad_norms: {k: norm} — norma média do gradiente para cada distância k.
        output_path: Caminho para salvar.
        title: Título do gráfico.
    """
    ks = sorted(grad_norms.keys())
    norms = [grad_norms[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(ks, norms, 'o-', color='#9C27B0', linewidth=2, markersize=4)
    ax.set_xlabel('Distância temporal k (frames)')
    ax.set_ylabel(r'$\|\partial L_t / \partial h_{t-k}\|$ (escala log)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)


def plot_framerate_stress(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = 'Teste de Estresse — IDF1 vs. Framerate',
) -> None:
    """Plota IDF1 degradation vs. framerate reduction.

    Args:
        results: {'1x': {'idf1': float}, '1/2x': {...}, '1/5x': {...}}
        output_path: Caminho para salvar.
        title: Título.
    """
    labels = sorted(results.keys())
    idf1_values = [results[l]['idf1'] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(labels)), idf1_values, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Framerate')
    ax.set_ylabel('IDF1')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)


def plot_ablation_results(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    title: str = 'Ablação Eixo 1 — Célula Recorrente × TBPTT',
) -> None:
    """Plota resultados de ablação como barras agrupadas.

    Args:
        results: {cell_type: {tbptt: {'idf1_mean': float, 'idf1_std': float}}}
        output_path: Caminho para salvar.
        title: Título.
    """
    cell_types = sorted(results.keys())
    tbptt_values = sorted(results[cell_types[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(cell_types))
    width = 0.3
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    for i, tbptt in enumerate(tbptt_values):
        means = [results[ct][tbptt]['idf1_mean'] for ct in cell_types]
        stds = [results[ct][tbptt]['idf1_std'] for ct in cell_types]
        offset = (i - len(tbptt_values) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=f'TBPTT={tbptt}',
               color=colors[i % len(colors)], alpha=0.8, capsize=3)

    ax.set_xlabel('Célula Recorrente')
    ax.set_ylabel('IDF1 (média ± std)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([ct.upper() for ct in cell_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
