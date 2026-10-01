# Plano de implementação — Isaías

## Escopo

Implementar `src/nms.py`, `src/detection/detector.py`, `src/tracking/tracker.py` e `src/tracking/temporal_model.py`, conforme a divisão em `plano.md`. Os quatro módulos foram implementados em 2026-10-01; veja a validação abaixo. Henrique cuida de associação, métricas e dados sintéticos; a integração dos entrypoints e os experimentos são trabalho conjunto.

## 1. Fixar os contratos antes de implementar

- Usar caixas em pixels `[x1, y1, x2, y2]` nos detectores e trackers; converter para `[cx, cy, w, h]` normalizados apenas na interface com a rede.
- Definir uma única convenção de cor no detector. O parser de imagens usa OpenCV; verificar a conversão BGR → RGB antes da inferência.
- Definir estado recorrente: tensor para RNN/GRU e tupla `(h, c)` para LSTM. Ajustar as anotações atualmente limitadas a tensor e importar `Optional`, que está ausente.
- Especificar se tracks sem observação são reportados pela baseline e pelo temporal, quando `hits` aumenta e como contar `max_age`. Validar os limites em testes.
- Fixar a semântica de `predict_single(hidden)`: a projeção do estado prevê a próxima caixa; o avanço recorrente precisa consumir uma observação ou predição. Evitar consumir o mesmo frame duas vezes.

## 2. NMS autoral

Ordenar por confiança, manter a caixa de maior score e remover caixas cuja IoU ultrapasse o limiar. Usar NumPy, tratar entrada vazia e áreas nulas, e retornar índices inteiros das caixas originais.

**Validação:** os cinco testes existentes em `tests/test_nms.py`, mais casos de igualdade ao limiar e caixas degeneradas.

```bash
python -m pytest tests/test_nms.py -v
```

## 3. Provedores de detecção

Implementar primeiro `MOT17DetectionLoader`, reutilizando `load_detections(seq_path)`, filtrando confiança e retornando lista vazia em frames sem detecção. Depois implementar Faster R-CNN pré-treinado, modo de avaliação, inferência sem gradientes, seleção da classe pessoa e NMS autoral.

**Ponto de atenção:** revisar o NMS interno do Faster R-CNN para compatibilidade com a proibição de `torchvision.ops.nms`; aplicar um NMS adicional na saída não demonstra conformidade por si só.

**Validação:** arquivo MOT mínimo temporário, filtro de confiança, frames vazios e detector simulado para testar classe, cor e coordenadas sem baixar pesos.

## 4. BaselineTracker

Implementar associação por IoU usando os contratos de Henrique, nascimento com IDs únicos, confirmação por `min_hits`, envelhecimento, remoção e `reset()`. Não duplicar os algoritmos de associação.

**Validação:** nascimento, continuidade de ID, ausência de detecções, confirmação, morte no limite de `max_age`, reset e matching greedy/Hungarian. A integração real depende da implementação de associação.

## 5. MotionRNN e treino

Implementar RNN/LSTM/GRU com projeção linear para quatro coordenadas. Extrair trajetórias ordenadas por frame, dividir em segmentos consecutivos para não tratar lacunas como um único passo e normalizar pelas dimensões reais da sequência.

Treinar com pares deslocados: entrada no instante `t`, alvo em `t+1`. Usar Smooth L1, gradient clipping e TBPTT, destacando o estado entre janelas e reiniciando-o entre trajetórias independentes.

**Validação:** formas de entrada/saída, estados das três células, conversão de coordenadas e redução da loss em movimento sintético simples. Executar primeiro em CPU; treino completo vem depois.

## 6. TemporalTracker

Prever a caixa antes do matching, atualizar o estado com a observação quando houver associação e usar a predição durante oclusão. Manter estado separado por identidade, mesma normalização do treino e inferência sem acumular grafos de gradiente.

**Validação:** movimento conhecido com modelo simulado, oclusão curta, recuperação de ID, descarte de tracks antigos e independência dos estados.

## 7. Integração e entregas

- Corrigir `train.py`: ele converte o GT em listas de dicionários, enquanto `extract_training_sequences` recebe `{track_id: {frame_id: bbox}}`. Passar o formato documentado e as dimensões reais.
- Corrigir `evaluate.py`: o split retorna uma tupla, não um dicionário; as funções do parser recebem `seq_path`; o loader recebe caminho e limiar, não raiz e sequência.
- Rodar a suíte completa e separar falhas dos módulos ainda pendentes de Henrique.
- Validar baseline nas sequências 09/10; treinar apenas em 02/04/05/11/13. Comparar temporal e baseline com as mesmas detecções e configurações.
- Depois executar ablações, galeria, horizonte de memória, estresse de framerate e notebook em conjunto. Inspecionar os scripts antes de assumir que produzem todos os resultados planejados.
- Registrar seeds, configurações, métricas e assistência de IA em `AI_LOG.md`; manter datasets e pesos fora do Git.

## Critério de conclusão individual

Os quatro módulos implementados, testes de comportamento passando, interfaces integradas e uma execução curta reproduzível de baseline e temporal. Melhoria de IDF1 é uma hipótese a medir; não deve ser presumida antes dos experimentos.


## Estado após implementação — 2026-10-01

- Concluídos: NMS, loader MOT17, wrapper Faster R-CNN, baseline, temporal,
  RNN/LSTM/GRU, extração de trajetórias consecutivas e treinamento com TBPTT.
- Corrigidas as interfaces de treino/avaliação e a leitura de métricas nas ablações.
- 37 testes passaram em CPU, incluindo execução CLI com checkpoint LSTM e
  inferência Faster R-CNN com o NMS nativo bloqueado. Dados e pesos desse teste
  são sintéticos/aleatórios; não demonstram qualidade de detecção real.
- Pendentes por ausência de dados: baseline e temporal em MOT17 real, treino final
  e comparação de IDF1. Ablações completas, galeria, horizonte de memória, estresse,
  notebook e apresentação continuam como etapas conjuntas posteriores.
