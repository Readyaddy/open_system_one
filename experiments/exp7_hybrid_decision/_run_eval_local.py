import sys, types, importlib.machinery, os
def _stub():
    if 'torchvision' in sys.modules: return
    fake_tv = types.ModuleType('torchvision')
    fake_tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', loader=None)
    fake_tv.__version__ = '0.0.0'
    fake_transforms = types.ModuleType('torchvision.transforms')
    fake_transforms.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms', loader=None)
    class InterpolationMode:
        NEAREST='nearest'; NEAREST_EXACT='nearest_exact'; BOX='box'
        BILINEAR='bilinear'; HAMMING='hamming'; BICUBIC='bicubic'; LANCZOS='lanczos'
    fake_transforms.InterpolationMode = InterpolationMode
    fake_tv.transforms = fake_transforms
    fake_io = types.ModuleType('torchvision.io')
    fake_io.__spec__ = importlib.machinery.ModuleSpec('torchvision.io', loader=None)
    fake_tv.io = fake_io
    sys.modules['torchvision'] = fake_tv
    sys.modules['torchvision.transforms'] = fake_transforms
    sys.modules['torchvision.io'] = fake_io
_stub()
sys.path.insert(0, os.path.dirname(__file__))
from eval import main
if __name__ == '__main__':
    main()
