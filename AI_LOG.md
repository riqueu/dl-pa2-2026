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
