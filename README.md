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

## 5. Inferência Arbitrária e Checkpoint

- **Notebook de Inferência:** [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb) atua como vitrine técnica do projeto (projetado para a apresentação de 15 minutos). Contém a função autossuficiente `predict_sequence(seq_path, ...)` que recebe qualquer sequência do MOT17 e gera a tira de keyframes e o player interativo JavaScript com identidades coloridas consistentes e métricas, sem retreinar.
- **Checkpoint Oficial (Trilha A - MotionRNN):**
  - Diferente do PA1 (onde a U-Net pesava ~99 MB e exigia hospedagem externa em GitHub Releases), a rede recorrente de movimento deste trabalho opera sobre coordenadas compactas ($[c_x, c_y, w, h]$), totalizando apenas ~14 mil parâmetros e **56 KB**.
  - Por ser ultraleve, o checkpoint oficial [`checkpoints/motion_gru.pth`](checkpoints/motion_gru.pth) já está **versionado diretamente no repositório**, garantindo execução imediata (*clone-and-run*) sem necessidade de downloads manuais ou dependência de links externos.
  - **Integridade Criptográfica (SHA-256):** `d8dca843a976b37fccbdb7c04d82fdde7200f47a6a47059203f19189d0aa53b0`.

---

## 6. Registro de IA e Planejamento

- [AI_LOG.md](AI_LOG.md): Registro de uso de ferramentas de IA.
- [docs/plano.md](docs/plano.md): Plano de implementação e divisão de tarefas.

## 7. Implementação e validação da parte de Isaías

NMS, provedores de detecção, baseline, tracker temporal e MotionRNN estão implementados.
A rede suporta RNN/LSTM/GRU, estados independentes por track e treino com TBPTT,
Smooth L1 e gradient clipping. Treino e inferência usam as dimensões reais da sequência.
O detector espera RGB uint8; o parser de imagens já entrega RGB.

```bash
python -m pytest tests/ -v
```

A suíte cobre ciclo de vida dos tracks, oclusão simulada, normalização, células
recorrentes e uma execução de treino/avaliação pela CLI em MOT17 mínimo temporário.
O teste de Faster R-CNN usa pesos aleatórios, sem downloads, e bloqueia o NMS nativo:
a implementação substitui as chamadas de RPN e RoI pelo NMS autoral somente na
instância do detector. Esse adaptador depende dos métodos internos do torchvision;
rode esse teste ao atualizar a biblioteca. Os pesos pré-treinados são obtidos pelo
wrapper na primeira utilização e ainda precisam ser avaliados em imagens reais.

Na baseline, tracks perdidos ficam disponíveis para reassociação, mas apenas os
observados no frame são reportados. No temporal, tracks confirmados podem ser
reportados com caixas preditas durante até `max_age` atualizações sem observação.
`hits` conta detecções associadas, incluindo o nascimento; `age` conta atualizações.
No teste de framerate, cada frame amostrado corresponde a uma atualização e as
métricas consideram somente esses frames.

A avaliação temporal exige checkpoint existente. Para arquiteturas diferentes,
passe também `--cell_type`, `--hidden_dim` e `--num_layers` correspondentes ao treino.
Os experimentos reais dependem do MOT17 em `data/`; galeria e análise de horizonte
em `scripts/run_failure_gallery.py` ainda são esboços. O script de estresse existente
ainda contém uma métrica placeholder e não deve ser usado como evidência experimental.
