# NeurOCAST_BiasCorrection

Applying NeurOCAST for bias correction of STOFS-2D-Global total water level forecast Guidance.


# Downloading and installing packages locally:

```
git clone git@github.com:AtiehAlipour-NOAA/NeurOCAST_BiasCorrection.git
cd NeurOCAST_BiasCorrection
pip install -e .
python patch_nemo.py
```

## Usage

```python
from neurocast import NeurOCAST
model = NeurOCAST()