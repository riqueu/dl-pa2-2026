import numpy as np
import pytest
import torch

from src.tracking.temporal_model import MotionRNN, extract_training_sequences, train_motion_model


@pytest.mark.parametrize('cell', ['rnn', 'lstm', 'gru'])
def test_shapes_online_and_training(cell):
    torch.manual_seed(42)
    model = MotionRNN(hidden_dim=8, num_layers=2, cell_type=cell)
    sequence = torch.rand(5, 4)
    predictions, hidden = model(sequence)
    prediction, unchanged = model.predict_single(hidden)
    np.testing.assert_allclose(prediction, predictions[-1].detach().numpy(), atol=1e-6)
    assert unchanged is hidden
    assert predictions.shape == (5, 4)
    initial = model.init_hidden(3)
    output, _ = model(torch.rand(5, 3, 4), initial)
    assert output.shape == (5, 3, 4)
    history = train_motion_model(model, [sequence, sequence.clone()], epochs=2,
                                 tbptt_len=2, batch_size=2, device='cpu')
    assert len(history['loss']) == 2
    assert np.isfinite(history['loss']).all()


def test_extraction_preserves_gaps_and_real_dimensions():
    box = np.array([10, 20, 30, 60])
    tracks = {1: {5: box, 2: box, 1: box, 6: box}, 2: {1: box}}
    sequences = extract_training_sequences(tracks, min_length=2, img_width=100, img_height=200)
    assert len(sequences) == 2
    for sequence in sequences:
        np.testing.assert_allclose(sequence.numpy(), [[.2, .2, .2, .2]] * 2)


def test_learns_simple_motion():
    torch.manual_seed(7)
    model = MotionRNN(hidden_dim=12)
    t = torch.arange(14) / 100
    sequence = torch.stack((.2 + t, .3 + t, torch.full_like(t, .1), torch.full_like(t, .2)), dim=1)
    history = train_motion_model(model, [sequence], epochs=20, lr=.01, tbptt_len=4, device='cpu')
    assert history['loss'][-1] < history['loss'][0] / 2
