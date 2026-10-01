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

    Returns:
        frames: Array uint8 (n_frames, img_size, img_size, 3).
        gt_tracks: Dict {track_id: {frame_id: bbox_xyxy}}.
        gt_per_frame: Dict {frame_id: [{'track_id': int, 'bbox': np.ndarray, 'visibility': float}]}.
    """
    rng = np.random.RandomState(seed)

    frames = np.zeros((n_frames, img_size, img_size, 3), dtype=np.uint8)
    gt_tracks: Dict[int, Dict[int, np.ndarray]] = {i + 1: {} for i in range(n_objects)}
    gt_per_frame: Dict[int, List[Dict]] = {f + 1: [] for f in range(n_frames)}

    # Propriedades de cada objeto
    depths = rng.permutation(n_objects)  # Maior profundidade = renderizado depois (na frente)
    render_order = np.argsort(depths)

    # Semi-eixos das elipses (raios horizontal e vertical)
    axes = rng.randint(6, 18, size=(n_objects, 2))
    # Cores distintas
    colors = rng.randint(60, 240, size=(n_objects, 3)).tolist()
    # Posições iniciais [x, y]
    positions = rng.uniform(25, img_size - 25, size=(n_objects, 2)).astype(np.float32)
    # Velocidades [vx, vy]
    velocities = rng.uniform(-4.0, 4.0, size=(n_objects, 2)).astype(np.float32)

    for f in range(n_frames):
        frame_id = f + 1

        # Canvas base com contraste variável
        contrast = rng.uniform(0.85, 1.15)
        base = np.full((img_size, img_size, 3), int(np.clip(120 * contrast, 40, 200)), dtype=np.uint8)

        # Atualiza posições e rebate nas bordas
        for obj_idx in range(n_objects):
            positions[obj_idx] += velocities[obj_idx]
            ax, ay = axes[obj_idx]

            # Bouncing
            if positions[obj_idx, 0] - ax <= 0:
                positions[obj_idx, 0] = ax
                velocities[obj_idx, 0] *= -1.0
            elif positions[obj_idx, 0] + ax >= img_size:
                positions[obj_idx, 0] = img_size - ax
                velocities[obj_idx, 0] *= -1.0

            if positions[obj_idx, 1] - ay <= 0:
                positions[obj_idx, 1] = ay
                velocities[obj_idx, 1] *= -1.0
            elif positions[obj_idx, 1] + ay >= img_size:
                positions[obj_idx, 1] = img_size - ay
                velocities[obj_idx, 1] *= -1.0

        # Renderiza elipses em ordem de profundidade (back-to-front / z-buffer)
        for obj_idx in render_order:
            cx, cy = int(positions[obj_idx, 0]), int(positions[obj_idx, 1])
            ax, ay = int(axes[obj_idx, 0]), int(axes[obj_idx, 1])
            color = colors[obj_idx]

            cv2.ellipse(base, (cx, cy), (ax, ay), 0, 0, 360, color, -1)

            # Bounding box delimitador exato da elipse
            x1 = float(np.clip(cx - ax, 0, img_size))
            y1 = float(np.clip(cy - ay, 0, img_size))
            x2 = float(np.clip(cx + ax, 0, img_size))
            y2 = float(np.clip(cy + ay, 0, img_size))
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            track_id = obj_idx + 1
            gt_tracks[track_id][frame_id] = bbox
            gt_per_frame[frame_id].append({
                'track_id': track_id,
                'bbox': bbox,
                'visibility': 1.0,
            })

        # Adiciona ruído gaussiano
        noise = rng.randn(img_size, img_size, 3) * (noise_std * 255.0)
        frame_noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames[f] = frame_noisy

    return frames, gt_tracks, gt_per_frame


def degrade_detections(
    gt_per_frame: Dict[int, List[Dict]],
    drop_rate: float = 0.1,
    noise_std: float = 3.0,
    fp_rate: float = 0.05,
    img_size: int = 128,
    seed: int = 42,
) -> Dict[int, List[Dict]]:
    """Simulador de detector imperfeito sobre as anotações do GT."""
    rng = np.random.RandomState(seed)
    degraded: Dict[int, List[Dict]] = {}

    for frame_id, objects in gt_per_frame.items():
        degraded[frame_id] = []
        for obj in objects:
            # Dropa p% das detecções (falsos negativos)
            if rng.rand() < drop_rate:
                continue

            bbox = obj['bbox'].copy()
            # Ruído nas coordenadas
            bbox += rng.randn(4).astype(np.float32) * noise_std
            bbox[0] = float(np.clip(bbox[0], 0, img_size))
            bbox[1] = float(np.clip(bbox[1], 0, img_size))
            bbox[2] = float(np.clip(bbox[2], 0, img_size))
            bbox[3] = float(np.clip(bbox[3], 0, img_size))

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            degraded[frame_id].append({
                'bbox': bbox,
                'confidence': float(rng.uniform(0.6, 0.98)),
            })

        # Falsos positivos
        num_fp = rng.poisson(fp_rate * max(1, len(objects)))
        for _ in range(num_fp):
            w = rng.uniform(10, 30)
            h = rng.uniform(10, 30)
            x1 = rng.uniform(0, img_size - w)
            y1 = rng.uniform(0, img_size - h)
            bbox_fp = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
            degraded[frame_id].append({
                'bbox': bbox_fp,
                'confidence': float(rng.uniform(0.1, 0.5)),
            })

    return degraded


def create_metric_edge_cases() -> List[Tuple[Dict, Dict, str]]:
    """Cria os 3 cenários de teste manuais exigidos pelo edital para validação das métricas."""
    # GT de base: 2 tracks ao longo de 10 frames
    gt_tracks = {
        1: {f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(1, 11)},
        2: {f: np.array([50.0, 50.0, 65.0, 65.0], dtype=np.float32) for f in range(1, 11)},
    }

    # 1. Predição perfeita
    pred_perfect = {
        1: {f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(1, 11)},
        2: {f: np.array([50.0, 50.0, 65.0, 65.0], dtype=np.float32) for f in range(1, 11)},
    }

    # 2. IDs invertidos no frame 6
    pred_swapped = {
        1: {f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(1, 6)},
        2: {f: np.array([50.0, 50.0, 65.0, 65.0], dtype=np.float32) for f in range(1, 6)},
    }
    # Troca de identificador
    pred_swapped[1].update({f: np.array([50.0, 50.0, 65.0, 65.0], dtype=np.float32) for f in range(6, 11)})
    pred_swapped[2].update({f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(6, 11)})

    # 3. Trajetórias divididas com gap temporal
    # Track 1 coberto pelo Pred 1 (frames 1-4) e Pred 3 (frames 7-10), gap nos frames 5-6
    pred_split = {
        1: {f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(1, 5)},
        2: {f: np.array([50.0, 50.0, 65.0, 65.0], dtype=np.float32) for f in range(1, 11)},
        3: {f: np.array([10.0, 10.0, 25.0, 25.0], dtype=np.float32) for f in range(7, 11)},
    }

    return [
        (gt_tracks, pred_perfect, "Predição perfeita: pred == gt -> IDF1 ≈ 1.0, ID switches = 0"),
        (gt_tracks, pred_swapped, "IDs trocados no meio da sequência -> ID switches >= 2"),
        (gt_tracks, pred_split, "Trajetória dividida com gap temporal -> fragmentations >= 1"),
    ]
