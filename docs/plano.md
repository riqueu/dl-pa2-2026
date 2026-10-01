# Plano Definitivo de Ataque — PA2: Multi-Object Tracking (MOT17)

> **Disciplina:** Aprendizado Profundo (FGV EMAp — 2026/2)  
> **Entrega & Apresentação:** 02/10/2026 às 23h59  
> **Dupla:** Henrique Coelho Beltrão & Isaias Gouvêa Gonçalves  
> **Hardware Principal:** NVIDIA GeForce RTX 4070  

---

## 1. Decisões Estratégicas Fechadas (Trilhas & Eixos)

Após análise técnica de viabilidade, tempo de convergência e alinhamento com os requisitos do edital, as escolhas foram fixadas:

| Componente | Escolha Oficial | Justificativa |
| :--- | :---: | :--- |
| **Parte 2 (Trilha)** | **Trilha A — RNN para Movimento** | Modela dinâmicas de trajetória espacial ($[c_x, c_y, w, h]$). Convergência rápida na RTX 4070, visualização intuitiva e integração direta com o matching por IoU da Parte 1. |
| **Parte 3 (Ablação)** | **Eixo 1 — Célula Recorrente** | Compara SimpleRNN vs. LSTM vs. GRU sob diferentes janelas de BPTT truncado ($T \in \{8, 32\}$) ao longo de 3 sementes ($42, 123, 456$). Produz gráficos limpos e resposta direta sobre vanishing gradient. |
| **Parte 5 (Estresse)** | **Framerate Drop** | Avalia a robustez do rastreador sob saltos temporais severos ($1\times$, $1/2\times$ e $1/5\times$), testando a capacidade de extrapolação da RNN quando $\Delta t$ aumenta. |
| **Detecções MOT17** | **SDP (Scale-Dependent Pooling)** | Maior taxa de recall e precisão entre os detectores públicos, gerando um baseline consistente para evidenciar os ganhos temporais. |
| **Split por Sequência** | **Treino:** `02, 04, 05, 11, 13`<br>**Validação:** `09, 10` | Atende à regra mandatória do edital de separar sequências inteiras sem vazamento temporal. `MOT17-09` e `10` são ideais para validação. |

---

## 2. Estado Atual do Repositório (Base & Infraestrutura)

Para otimizar o tempo e focar o esforço nas partes conceituais de Deep Learning, a infraestrutura básica foi padronizada:

*   **Infraestrutura pronta (Boilerplate):**
    *   `src/data/mot17.py`: parser oficial de sequências MOT17 (`seqinfo.ini`, detecções públicas, ground truth filtrado para pedestres).
    *   `src/visualization.py`: desenho de bboxes com colorização consistente de identidades, strips visuais de falha e funções de plot.
    *   `requirements.txt` e estrutura de pastas em `outputs/`, `checkpoints/`, `notebooks/`.
*   **Suíte de Testes para TDD (`tests/`):**
    *   23 testes unitários cobrindo associação, métricas, NMS e dados sintéticos.
*   **Módulos com Contratos Estritos (Aguardando implementação da dupla):**
    *   Todos os módulos algorítmicos essenciais foram declarados com contratos de tipos, docstrings e `raise NotImplementedError`.

---

## 3. Divisão de Tarefas em Paralelo

A divisão garante isolamento total de arquivos entre as branches para evitar conflitos de merge:

```
                  ┌────────────────────────────────────────┐
                  │     main (infraestrutura e contratos)  │
                  └───────────────────┬────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   feature/data-metrics-association        feature/detection-tracking-model
   (Membro 1 — Henrique)                  (Membro 2 — Isaias)
   • src/metrics.py                       • src/nms.py
   • src/data/synthetic.py                • src/detection/detector.py
   • src/tracking/association.py          • src/tracking/tracker.py
   • Testes: tests/test_metrics.py,       • src/tracking/temporal_model.py
     test_synthetic.py,                   • Testes: tests/test_nms.py
     test_association.py                  • Validação de inferência do modelo
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        Integração & Treino Conjunto
                        • train.py & evaluate.py
                        • scripts/run_ablation_eixo1.py (18 runs)
                        • scripts/run_failure_gallery.py
                        • scripts/run_stress_framerate.py
                        • notebooks/inferencia.ipynb
                        • Slides da Apresentação
```

---

### Membro 1 (Henrique) — Dados Sintéticos, Métricas & Associação

*   **Branch de Trabalho:** `feature/data-metrics-association`
*   **Módulos Sob Sua Responsabilidade:**
    1.  `src/metrics.py`:
        *   `compute_bbox_iou` e `compute_iou_matrix` (vetorizada).
        *   `compute_idf1`: bipartite matching global ótimo via `linear_sum_assignment` da SciPy.
        *   `count_id_switches`: contagem temporal de trocas de identidade.
        *   `count_fragmentations`: transições de track rastreado $\to$ perdido.
    2.  `src/data/synthetic.py`:
        *   `generate_synthetic_video`: 5 a 15 elipses, 128x128, ruído e contraste, com ordenação por profundidade real (z-buffer) para oclusões.
        *   `degrade_detections`: simulação de detector imperfeito (drop de detecções, ruído gaussiano nas caixas, injeção de falsos positivos).
        *   `create_metric_edge_cases`: os 3 casos do edital (predição perfeita, IDs invertidos, tracks divididos).
    3.  `src/tracking/association.py`:
        *   `greedy_matching`: casamento guloso por IoU decrescente.
        *   `hungarian_matching`: casamento ótimo via Húngaro com limiar de custo.
*   **Comando de Validação (TDD):**
    ```bash
    python -m pytest tests/test_metrics.py tests/test_synthetic.py tests/test_association.py -v
    ```

---

### Membro 2 (Isaias) — NMS, Provedores de Detecção & Pipeline Recorrente

*   **Branch de Trabalho:** `feature/detection-tracking-model`
*   **Módulos Sob Sua Responsabilidade:**
    1.  `src/nms.py`:
        *   `nms`: implementação do zero com NumPy (proibido usar `torchvision.ops.nms`).
    2.  `src/detection/detector.py`:
        *   `MOT17DetectionLoader`: carregador de detecções pré-computadas (SDP).
        *   `TorchvisionDetector`: wrapper de Faster R-CNN pré-treinado filtrando pedestres e aplicando o NMS autoral.
    3.  `src/tracking/tracker.py`:
        *   `TrackState`: representação do estado individual de cada trajetória.
        *   `BaselineTracker`: ciclo de vida (birth, hits $\ge 3$, death após `max_age=30`) com matching por IoU.
        *   `TemporalTracker`: herda do baseline, calcula IoU contra as caixas preditas pela RNN e mantém o estado em coasting durante oclusões.
    4.  `src/tracking/temporal_model.py`:
        *   `MotionRNN`: arquitetura PyTorch com suporte a `cell_type in ['rnn', 'lstm', 'gru']`.
        *   `extract_training_sequences`: extrai e normaliza trajetórias do GT do MOT17.
        *   `train_motion_model`: loop com Truncated BPTT (`tbptt_len=16`), gradient clipping e Smooth L1 Loss.
*   **Comando de Validação (TDD):**
    ```bash
    python -m pytest tests/test_nms.py -v
    ```

---

## 4. Cronograma Marco a Marco de Execução

### Marco 1: Implementação dos Componentes & TDD (Individual)
*   **Henrique:** Implementa `src/metrics.py`, `src/data/synthetic.py` e `src/tracking/association.py`. Todos os 18 testes unitários correspondentes passam no `pytest`.
*   **Isaias:** Implementa `src/nms.py`, `src/detection/detector.py`, `src/tracking/tracker.py` e `src/tracking/temporal_model.py`. Os 5 testes unitários de NMS passam.

### Marco 2: Merge na `main` & Smoke Test da Baseline (Parte 0 e Parte 1)
*   Merge das duas branches sem conflitos de arquivo.
*   **Validação da Parte 0:** Rodar os 3 edge cases manuais das métricas e verificar que IDF1 = 1.0 no perfeito e acusa trocas no invertido.
*   **Execução da Parte 1:**
    ```bash
    # Avaliar baseline nas sequências de validação (MOT17-09 e MOT17-10)
    python evaluate.py --mode baseline --det_type SDP --split val --output-dir outputs/part1_baseline
    ```
*   Gerar o gráfico de quantificação de falhas vs. densidade de pedestres.

### Marco 3: Treinamento do Modelo Temporal & Ganho na Validação (Parte 2)
*   Treinar a MotionRNN com célula GRU na RTX 4070:
    ```bash
    python train.py --cell_type gru --hidden_dim 64 --epochs 40 --lr 1e-3 --checkpoint checkpoints/motion_gru.pth --out runs/part2_gru
    ```
*   Avaliar e comparar diretamente contra a baseline:
    ```bash
    python evaluate.py --mode temporal --checkpoint checkpoints/motion_gru.pth --split val --output-dir outputs/part2_temporal
    ```
*   **Meta de Sucesso:** O modelo temporal deve superar o baseline em IDF1 e reduzir as trocas de identidade durante oclusões parciais.

### Marco 4: Execução das Ablações do Eixo 1 (Parte 3)
*   Executar as 18 configurações (3 células $\times$ 2 comprimentos TBPTT $\times$ 3 sementes):
    ```bash
    python scripts/run_ablation_eixo1.py
    ```
*   O script consolida média $\pm$ desvio-padrão amostral (`ddof=1`) em `outputs/part3_ablation/summary.json` e plota o gráfico de barras agrupadas.

### Marco 5: Horizonte de Memória & Galeria de Falhas (Parte 4)
*   **Horizonte Analítico:** Calcular a norma do gradiente $\partial L_t / \partial h_{t-k}$ para $k \in [1, 40]$ na SimpleRNN vs. GRU, plotando a curva logarítmica demonstrando o decaimento exponencial (vanishing gradient).
*   **Horizonte Empírico:** Medir a distribuição de frames de oclusão que a GRU consegue manter a identidade sem perder o objeto.
*   **Galeria:** Isolar 3 tiras visuais de falhas reais com diagnóstico e aplicar 1 correção ajustada (ex.: calibrar limiar de coasting).

### Marco 6: Teste de Estresse de Framerate (Parte 5)
*   Avaliar o modelo final sob $1\times$, $1/2\times$ e $1/5\times$ do framerate sem retreinar:
    ```bash
    python scripts/run_stress_framerate.py --checkpoint checkpoints/motion_gru.pth
    ```
*   Gerar a curva de degradação do IDF1.

### Marco 7: Entregáveis Finais & Apresentação
*   **Notebook:** Preencher `notebooks/inferencia.ipynb` com o pipeline completo (recebe caminho de sequência e salva vídeo renderizado com cores consistentes e contagem final).
*   **AI Log:** Atualizar `AI_LOG.md` registrando as interações.
*   **Slides da Apresentação:** Montar os slides com os gráficos de `outputs/` (a avaliação oficial é 100% oral baseada na apresentação).

---

## 5. Comandos de Referência Rápida

```bash
# Ativar ambiente virtual
source /home/rique/Documents/fgv/6o/DL/assignments/dl-pa1-2026/venv/bin/activate

# Rodar todos os testes unitários
python -m pytest tests/ -v

# Treinar modelo temporal padrão
python train.py --cell_type gru --epochs 40 --checkpoint checkpoints/motion_gru.pth

# Avaliar modelo temporal na validação
python evaluate.py --mode temporal --checkpoint checkpoints/motion_gru.pth --split val
```
