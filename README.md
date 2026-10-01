# PA2 - Multi-Object Tracking com Memória Temporal (RNN)

Programming Assignment 2 da disciplina de Aprendizado Profundo (FGV EMAp).
Rastreamento multi-objeto (MOT) sobre o benchmark MOT17 utilizando associação autoral, modelo temporal recorrente (Trilha A - RNN para Movimento) e ablações sobre células recorrentes.

## Autores
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral

O projeto implementa **do zero** um sistema de tracking multi-objeto com:
- **Parte 0:** Testes sintéticos (gerador de vídeos + detector simulator + validação de métricas)
- **Parte 1:** Baseline frame-by-frame (detecções SDP + torchvision FRCNN, associação por IoU)
- **Parte 2:** Memória temporal via RNN de movimento (Trilha A - GRU prediz próximo bounding box)
- **Parte 3:** Ablação Eixo 1 (SimpleRNN vs LSTM vs GRU × TBPTT ∈ {8, 32} × 3 seeds)
- **Parte 4:** Galeria de falhas + horizonte de memória (gradiente ∂L_t/∂h_{t-k} + sobrevivência em oclusão)
- **Parte 5:** Teste de estresse por queda de framerate (1/2 e 1/5)

> **Proibições:** SORT/DeepSORT/ByteTrack, `motmetrics`/`TrackEval`, `torchvision.ops.nms`. Tudo é autoral.

---

## 2. Configuração do Ambiente

Python 3.10+ em ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Dados (MOT17)

1. Baixe o dataset [MOT17](https://motchallenge.net/data/MOT17/) e as labels:
```bash
# Imagens + detecções
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d data/

# Labels (ground truth)
wget https://motchallenge.net/data/MOT17Labels.zip
unzip MOT17Labels.zip -d data/
```

2. Estrutura esperada:
```
data/
├── MOT17/
│   ├── train/
│   │   ├── MOT17-02-SDP/
│   │   │   ├── det/det.txt
│   │   │   ├── gt/gt.txt
│   │   │   ├── img1/
│   │   │   └── seqinfo.ini
│   │   └── ...
│   └── test/
└── MOT17Labels/
    └── train/
```

**Split por sequência:**
- **Treino:** MOT17-02, MOT17-04, MOT17-05, MOT17-11, MOT17-13
- **Validação:** MOT17-09, MOT17-10

---

## 4. Execução Rápida

### 4.1. Comandos Principais (Edital)

- **Treinamento do modelo temporal (Trilha A - GRU):**
  ```bash
  python train.py --cell_type gru --hidden_dim 64 --epochs 50 --lr 1e-3 --tbptt_len 16 --checkpoint checkpoints/motion_gru.pth
  ```

- **Avaliação completa em sequências de validação:**
  ```bash
  python evaluate.py --checkpoint checkpoints/motion_gru.pth --det_type SDP --split val
  ```

### 4.2. Comandos Adicionais para Reprodução

```bash
# Parte 0 - Testes sintéticos e validação de métricas
python -m pytest tests/test_metrics.py tests/test_synthetic.py tests/test_nms.py -v

# Parte 1 - Baseline frame-by-frame (sem modelo temporal)
python evaluate.py --mode baseline --det_type SDP --split val --output-dir outputs/part1_baseline

# Parte 3 - Ablação Eixo 1 (3 células × 2 TBPTT × 3 seeds = 18 runs)
python scripts/run_ablation_eixo1.py

# Parte 4 - Galeria de falhas + horizonte de memória
python scripts/run_failure_gallery.py --checkpoint checkpoints/motion_gru.pth

# Parte 5 - Teste de estresse de framerate
python scripts/run_stress_framerate.py --checkpoint checkpoints/motion_gru.pth
```

---

## 5. Inferência Arbitrária

- **Notebook:** [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) recebe o caminho de uma sequência MOT17 e gera um vídeo com identidades coloridas consistentes e contagem de objetos únicos, sem retreinar.

---

## 6. Registro de IA e Planejamento

- [AI_LOG.md](AI_LOG.md): Registro de uso de ferramentas de IA.
- [docs/plano.md](docs/plano.md): Plano de implementação e divisão de tarefas.
