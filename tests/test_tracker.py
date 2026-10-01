import numpy as np
import pytest
import torch
from torch import nn

from src.tracking.tracker import BaselineTracker, TemporalTracker


def detection(x=0):
    return [{'bbox': np.array([x, 0, x + 10, 10], dtype=np.float32)}]


@pytest.mark.parametrize('matching', ['hungarian', 'greedy'])
def test_confirmation_death_and_reset(matching):
    tracker = BaselineTracker(min_hits=2, max_age=1, matching=matching)
    assert tracker.update(detection()) == []
    assert tracker.update(detection(1))[0].track_id == 1
    assert tracker.update([]) == []
    assert len(tracker.tracks) == 1
    tracker.update([])
    assert tracker.tracks == []
    tracker.update(detection())
    assert tracker.tracks[0].track_id == 2
    tracker.reset()
    assert tracker.next_id == 1 and tracker.tracks == []


class ConstantVelocity(nn.Module):
    def forward(self, observation, hidden=None):
        # Tracks advance 10 pixels per step in a 100-pixel-wide image.
        state = observation.clone()
        state[..., 0] += .1
        return state, state

    def predict_single(self, hidden):
        return hidden[0, 0].cpu().numpy(), hidden


def test_temporal_crosses_occlusion_without_reusing_observation():
    tracker = TemporalTracker(ConstantVelocity(), min_hits=1, max_age=2,
                              img_width=100, img_height=100, device='cpu')
    assert tracker.update(detection())[0].track_id == 1
    assert tracker.update(detection(10))[0].track_id == 1
    coast = tracker.update([])[0]
    np.testing.assert_allclose(coast.bbox, [20, 0, 30, 10], atol=1e-5)
    assert tracker.update(detection(30))[0].track_id == 1
    tracker.update([])
    tracker.update([])
    assert tracker.update([]) == []


def test_each_track_has_independent_hidden_state():
    tracker = TemporalTracker(ConstantVelocity(), min_hits=1, img_width=100,
                              img_height=100, device='cpu')
    tracks = tracker.update(detection() + detection(50))
    assert tracks[0].rnn_hidden is not tracks[1].rnn_hidden
    tracker.update([])
    np.testing.assert_allclose([t.bbox[0] for t in tracker.tracks], [10, 60], atol=1e-5)
