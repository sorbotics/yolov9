"""segment/predict.py and panoptic/predict.py must feed process_mask the prototype tensor.

In eval mode the Segment / DSegment heads return (pred, (x, mc, protos)) and Panoptic returns
(pred, (x, mc, protos, semantic)); the old `model(im)[:2]` unpacking handed the whole tuple to
process_mask, which crashed with "'list' object has no attribute 'shape'" as soon as one detection
survived NMS. CPU only, random weights, a few seconds.
"""
import importlib

import numpy as np
import pytest
import torch
from PIL import Image

from yolo9.models.yolo import DetectionModel

# P3/P4/P5 feature maps come from layers 2, 3, 4; the head is the only part that changes per case.
BACKBONE = "[[-1, 1, Conv, [8, 3, 2]], [-1, 1, Conv, [16, 3, 2]], [-1, 1, Conv, [16, 3, 2]], [-1, 1, Conv, [16, 3, 2]], [-1, 1, Conv, [16, 3, 2]]]"
HEADS = {
    "segment": ("segment", "[[[2, 3, 4], 1, Segment, [nc, 32, 16]]]"),
    "dsegment": ("segment", "[[[2, 3, 4, 4], 1, DSegment, [nc, 32, 16]]]"),  # extra input feeds the proto conv
    "panoptic": ("panoptic", "[[[2, 3, 4], 1, Panoptic, [nc, 8, 32, 16]]]"),
}


@pytest.mark.parametrize("case", HEADS)
def test_predict_runs_when_a_detection_is_found(case, tmp_path, monkeypatch):
    package, head = HEADS[case]
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(f"nc: 2\ndepth_multiple: 1.0\nwidth_multiple: 1.0\nanchors: 3\nbackbone: {BACKBONE}\nhead: {head}\n")
    torch.manual_seed(0)
    model = DetectionModel(str(cfg), ch=3, nc=2).eval()
    for seq in model.model[-1].cv3:  # make every anchor a confident class-0 detection, so NMS keeps some
        seq[-1].bias.data.fill_(10.0)

    class Backend:  # stands in for DetectMultiBackend, which needs a saved checkpoint
        def __init__(self, *args, **kwargs):
            self.stride, self.names, self.pt, self.fp16, self.device = 32, {0: "a", 1: "b"}, True, False, torch.device("cpu")

        def warmup(self, imgsz):
            pass

        def __call__(self, im, augment=False, visualize=False):
            with torch.no_grad():
                return model(im)

    predict = importlib.import_module(f"yolo9.{package}.predict")
    monkeypatch.setattr(predict, "DetectMultiBackend", Backend)
    img = tmp_path / "img.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(img)

    predict.run(weights="unused.pt", source=str(img), imgsz=(64, 64), conf_thres=0.5, project=tmp_path / "out", save_txt=True)

    labels = list((tmp_path / "out").glob("exp*/labels/img.txt"))
    assert labels and labels[0].read_text().strip(), "no segment written: no detection reached process_mask"
