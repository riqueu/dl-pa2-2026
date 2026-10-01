"""Script para execução das 18 ablações do Eixo 1 (Célula Recorrente).

Matriz Experimental (Parte 3):
- Células: SimpleRNN vs LSTM vs GRU (3)
- TBPTT length: T in {8, 32} (2)
- Sementes: {42, 123, 456} (3)
- Total: 3 x 2 x 3 = 18 execuções

Responsável pela execução: Membro 2 (Isaias) / Henrique na RTX 4070
"""

import itertools
import json
import os
import subprocess
import sys
import numpy as np


def main():
    cell_types = ['rnn', 'lstm', 'gru']
    tbptt_lens = [8, 32]
    seeds = [42, 123, 456]

    output_dir = 'outputs/part3_ablation'
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for ct, tbptt, seed in itertools.product(cell_types, tbptt_lens, seeds):
        run_name = f"{ct}_tbptt{tbptt}_seed{seed}"
        ckpt_path = f"checkpoints/ablation_{run_name}.pth"
        run_out = f"runs/ablation_{run_name}"
        eval_out = f"{output_dir}/{run_name}"

        print(f"\n==========================================")
        print(f"Executando ablação: Célula={ct.upper()}, TBPTT={tbptt}, Seed={seed}")
        print(f"==========================================")

        # 1. Treino
        train_cmd = [
            sys.executable, "train.py",
            "--cell_type", ct,
            "--tbptt_len", str(tbptt),
            "--seed", str(seed),
            "--epochs", "1",
            "--checkpoint", ckpt_path,
            "--out", run_out,
        ]
        subprocess.run(train_cmd, check=True)

        # 2. Avaliação
        eval_cmd = [
            sys.executable, "evaluate.py",
            "--mode", "temporal",
            "--checkpoint", ckpt_path,
            "--cell_type", ct,
            "--split", "val",
            "--output-dir", eval_out,
        ]
        subprocess.run(eval_cmd, check=True)

        # 3. Ler métricas reais salvas pelo evaluate.py
        summary_file = os.path.join(eval_out, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                metrics = json.load(f)
            idf1 = metrics['idf1']
            idsw = metrics['id_switches']
        else:
            raise FileNotFoundError(summary_file)

        results.append({
            "cell_type": ct,
            "tbptt_len": tbptt,
            "seed": seed,
            "idf1": idf1,
            "id_switches": idsw,
        })

    # Consolidar resumo em JSON
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTodas as 18 ablações foram concluídas com sucesso!")


if __name__ == "__main__":
    main()
