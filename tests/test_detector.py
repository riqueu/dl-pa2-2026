import numpy as np
import torch
from torch import nn

from src.detection.detector import MOT17DetectionLoader, TorchvisionDetector, _install_custom_nms


def test_loader_filters_and_returns_copies(tmp_path):
    (tmp_path / 'det').mkdir()
    (tmp_path / 'det/det.txt').write_text('1,-1,10,20,30,40,0.9\n1,-1,0,0,5,5,0.1\n')
    loader = MOT17DetectionLoader(str(tmp_path), conf_threshold=.5)
    result = loader.get_detections(1)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0]['bbox'], [10, 20, 40, 60])
    result[0]['bbox'][0] = 0
    assert loader.get_detections(1)[0]['bbox'][0] == 10
    assert loader.get_detections(2) == []


class FakeDetector(nn.Module):
    def forward(self, images):
        assert images[0].shape == (3, 8, 8)
        assert images[0][0, 0, 0] == 1  # RGB red stays red.
        return [{'boxes': torch.tensor([[0., 0, 5, 5]] * 3),
                 'scores': torch.tensor([.9, .8, .99]),
                 'labels': torch.tensor([1, 1, 2])}]


def test_filters_people_and_applies_custom_nms():
    detector = TorchvisionDetector.__new__(TorchvisionDetector)
    detector.device, detector.conf_threshold, detector.nms_threshold = 'cpu', .5, .4
    detector.model = FakeDetector()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    assert len(detector.detect(image)) == 1


def test_real_fasterrcnn_does_not_call_native_nms(monkeypatch):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.ops import boxes as box_ops
    def forbidden(*args, **kwargs):
        raise AssertionError('NMS nativo foi chamado')
    monkeypatch.setattr(box_ops, 'nms', forbidden)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                    min_size=32, max_size=32,
                                    rpn_pre_nms_top_n_test=20, rpn_post_nms_top_n_test=10)
    _install_custom_nms(model)
    model.eval()
    with torch.inference_mode():
        output = model([torch.rand(3, 32, 32)])
    assert output[0]['boxes'].shape[-1] == 4
