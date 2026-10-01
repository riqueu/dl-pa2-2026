import json
import os
from pathlib import Path
import subprocess
import sys


def test_train_and_evaluate_end_to_end(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / 'data'
    for number in ('02', '04', '05', '11', '13', '09', '10'):
        seq = data / 'MOT17/train' / f'MOT17-{number}-SDP'
        (seq / 'gt').mkdir(parents=True)
        (seq / 'det').mkdir()
        (seq / 'seqinfo.ini').write_text(
            '[Sequence]\nname=test\nimWidth=100\nimHeight=80\nseqLength=12\n')
        (seq / 'gt/gt.txt').write_text(''.join(
            f'{frame},1,20,20,10,20,1,1,1\n' for frame in range(1, 13)))
        (seq / 'det/det.txt').write_text(''.join(
            f'{frame},-1,20,20,10,20,.9\n' for frame in range(1, 13)))
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    def run(*args):
        result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / 'model.pth'
    run('train.py', '--data_root', str(data), '--device', 'cpu', '--epochs', '2',
        '--hidden_dim', '8', '--num_layers', '2', '--cell_type', 'lstm',
        '--tbptt_len', '3', '--batch_size', '2', '--checkpoint', str(checkpoint),
        '--out', str(tmp_path / 'train'))
    assert checkpoint.is_file()
    history = json.loads((tmp_path / 'train/history.json').read_text())
    assert len(history['loss']) == 2
    for mode in ('baseline', 'temporal'):
        out = tmp_path / mode
        run('evaluate.py', '--data_root', str(data), '--device', 'cpu',
            '--mode', mode, '--checkpoint', str(checkpoint), '--hidden_dim', '8',
            '--num_layers', '2', '--cell_type', 'lstm', '--min_hits', '1',
            '--subsample', '2', '--output-dir', str(out))
        summary = json.loads((out / 'summary.json').read_text())
        assert 0 <= summary['idf1'] <= 1
        assert (out / 'idf1.png').is_file()
        if mode == 'baseline':
            assert summary['idf1'] == 1.0  # GT must be filtered to sampled frames.
