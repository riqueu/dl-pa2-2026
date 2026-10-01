import numpy as np
import pytest
from src.nms import nms

def test_no_suppression():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    scores = np.array([0.9, 0.8])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert set(keep) == {0, 1}

def test_full_suppression():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]])
    scores = np.array([0.9, 0.95, 0.8])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 1
    assert keep[0] == 1

def test_partial_suppression():
    """Two overlapping boxes + one distant → lower threshold suppresses overlap."""
    boxes = np.array([[0, 0, 10, 10], [2, 2, 12, 12], [30, 30, 40, 40]])
    scores = np.array([0.9, 0.8, 0.85])
    # IoU between box 0 and 1 is ~0.47, so threshold 0.4 triggers suppression
    keep = nms(boxes, scores, iou_threshold=0.4)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep

def test_empty_input():
    boxes = np.empty((0, 4))
    scores = np.empty((0,))
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 0

def test_single_box():
    boxes = np.array([[0, 0, 10, 10]])
    scores = np.array([0.9])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 1
    assert keep[0] == 0
