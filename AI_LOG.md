# Registro de Uso de IA (AI_LOG)

Uso de ferramentas de IA como suporte de pair programming, planejamento e divisão de atividades da dupla (`docs/`), automação de scripts e revisão teórica. Todas as decisões técnicas, implementações e interpretações dos resultados foram validadas e dominadas pela dupla.

---

## 2026-10-01 — Implementação da parte de Isaías

Codex foi usado para implementar NMS NumPy, carregador MOT17, wrapper Faster R-CNN,
ciclo de vida dos trackers e modelo RNN/LSTM/GRU com TBPTT. A integração corrigiu os
contratos de `train.py` e `evaluate.py`, preservando os módulos de Henrique.
Foram adicionados testes de estados recorrentes, oclusão, detector e execução CLI.

Validação automatizada em CPU no ambiente local do PA1 (Python/PyTorch 2.14.0,
torchvision instalado): 37 testes passaram. O teste de Faster R-CNN usa pesos
aleatórios e proíbe chamadas ao NMS nativo. A integração CLI usa dados temporários
mínimos, não resultados oficiais do MOT17. O dataset e checkpoints reais estão
ausentes deste checkout; não houve treino completo, ablações nem validação de ganho
de IDF1 em sequências reais. A revisão e interpretação pela dupla seguem pendentes.

---

## 2026-10-01 — Planejamento Inicial e Depuração de Rastreamento (Henrique & Isaías)

A IA foi utilizada para estruturar o planejamento inicial do projeto (`docs/plano_pa2.md`), definindo interfaces modulares e a divisão de escopo entre os integrantes.

Na fase de integração e validação visual no notebook de inferência, a IA auxiliou na correção de bugs relacionado à visualização de vídeos no notebook e no diagnóstico e resolução de dois comportamentos no `TemporalTracker`:
1. **Caixas fantasmas (*ghost coasting*):** Identificação de trajetórias perdidas que continuavam sendo desenhadas na tela sem observações recentes, corrigida com filtro estrito de exibição ativa (`time_since_update <= max_coast`).
2. **Confinamento cinemático da RNN:** Diagnóstico de instabilidades pontuais no estado oculto da rede de movimento que causavam saltos abruptos de bounding boxes e explosão de novos IDs, estabilizado com *gating* de plausibilidade física (`max_shift`).

Todas as hipóteses, soluções e análises teóricas foram testadas via suíte automatizada (37/37 testes aprovados) e validadas pela dupla no MOT17.
