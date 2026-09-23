"""Wrapper for running train.py LOCALLY on this machine only.

This machine's torchvision install is broken in a way that blocks
`transformers` from importing ANY model class (`RuntimeError: operator
torchvision::nms does not exist`, a torch/torchvision version mismatch) --
confirmed unrelated to this project's own code, and worked around all
session via a fake-module stub. train.py itself is deliberately kept clean
of this (it runs fine as-is on Colab, where torchvision isn't broken), so
the stub is applied here, in a wrapper, instead of inside train.py.

Usage: identical to train.py, just invoke this file instead --
  python run_train_local.py --k_max 6 --freeze_backbone ...
"""
import sys
import types
import importlib.machinery


def _install_torchvision_stub():
    if "torchvision" in sys.modules:
        return  # already real or already stubbed
    fake_tv = types.ModuleType("torchvision")
    fake_tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    fake_tv.__version__ = "0.0.0"

    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)

    class InterpolationMode:
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest_exact"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    fake_transforms.InterpolationMode = InterpolationMode
    fake_tv.transforms = fake_transforms

    fake_io = types.ModuleType("torchvision.io")
    fake_io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)
    fake_tv.io = fake_io

    sys.modules["torchvision"] = fake_tv
    sys.modules["torchvision.transforms"] = fake_transforms
    sys.modules["torchvision.io"] = fake_io


_install_torchvision_stub()

import os
sys.path.insert(0, os.path.dirname(__file__))
from train import main

if __name__ == "__main__":
    main()
