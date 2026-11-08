# RF-DTAN Training Guide — Using Your Own Data

## 1. Installation

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install difw==0.0.29 tsai==0.2.24 tslearn
```

---

## 2. Data Format

RF-DTAN expects a DataLoader yielding `(X, y)` batches where:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X` | `(batch, k, signal_len)` | Time-series — **channels-first** |
| `y` | `(batch,)` | Integer class labels starting from `0` |

### Handling variable-length samples

The correct approach for variable-length signals is **NaN-padding**, not interpolation/resampling.
Resampling would impose a specific type of temporal stretching, which defeats the purpose of the framework.

**How it works inside the model:**
1. Pad each sample with `NaN` at the end to the length of the longest sample
2. `RFDTAN.forward()` auto-detects NaNs, builds a mask, and fills them with `0` for computation
3. The NaN mask is warped alongside the data so real vs. padded regions are always tracked
4. `Trainer` detects `has_nans` at init and switches to NaN-aware losses (`MSENANLoss`, `TripletNANLoss`) automatically

### Preprocessing script

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper.UCR_loader import UCRDataset

def prepare_data(samples, labels, batch_size=64, normalize=True):
    """
    Args:
        samples    : list of np.ndarray, each shape (n_i, k)
                     n_i varies per sample, k = number of channels
        labels     : list/array of int class labels, length = len(samples)
        batch_size : int
        normalize  : z-normalize per channel across the dataset (recommended)

    Returns:
        loader     : DataLoader yielding (X, y), X shape (batch, k, max_len)
        signal_len : int, max_len — pass this to RFDTAN as signal_len
    """
    k       = samples[0].shape[1]
    max_len = max(s.shape[0] for s in samples)

    padded = []
    for s in samples:
        n_i     = s.shape[0]
        pad     = max_len - n_i
        s_pad   = np.pad(s, ((0, pad), (0, 0)), constant_values=np.nan)  # (max_len, k)
        padded.append(s_pad.T)                                            # (k, max_len)

    X = np.stack(padded, axis=0).astype(np.float32)   # (N, k, max_len)
    y = np.array(labels, dtype=np.int64)

    if normalize:
        for ch in range(k):
            mu  = np.nanmean(X[:, ch, :])
            std = np.nanstd(X[:, ch, :]) + 1e-8
            X[:, ch, :] = (X[:, ch, :] - mu) / std

    dataset = UCRDataset(torch.tensor(X), torch.tensor(y))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader, max_len

# Example
# samples = [np.random.randn(120, 1), np.random.randn(95, 1), np.random.randn(110, 1)]
# labels  = [0, 0, 0]   # single class is fine
# train_loader, signal_len = prepare_data(samples, labels)
```

> If all samples already have the **same length**, no NaN-padding is needed — just stack
> directly into `(N, k, signal_len)` and skip the padding step.

---

## 3. Training Script

```python
import torch
from config import get_argparser
from models.RFDTAN import RFDTAN
from models.trainer import Trainer

train_loader, signal_len = prepare_data(samples_train, labels_train, batch_size=64)

parser = get_argparser()
args   = parser.parse_args("")

args.ICAE_loss          = True    # recommended — works for single class too
args.ICAE_triplets_loss = False   # needs >= 2 classes
args.WCSS_loss          = False
args.WCSS_triplets_loss = False   # needs >= 2 classes
args.smoothness_prior   = False
args.zero_boundary      = True
args.n_recurrences      = 4
args.batch_size         = 64
args.n_epochs           = 500
args.tess_size          = 16
args.lr                 = 0.0005
args.no_validation      = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RFDTAN(
    signal_len  = signal_len,   # max_len returned by prepare_data
    channels    = 1,            # k from your data
    tess        = args.tess_size,
    n_recurrence= args.n_recurrences,
    zero_boundary = args.zero_boundary,
    device      = device,
    backbone    = args.backbone,
)

trainer = Trainer(
    exp_name     = "my_experiment",
    model        = model,
    train_loader = train_loader,
    val_loader   = None,
    device       = device,
    args         = args,
)
trainer.train()
```

---

## 4. Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tess_size` | 16 | CPA partition — larger = more expressive warps |
| `n_recurrences` | 4 | Iterative alignment passes (no extra params) |
| `zero_boundary` | True | Forces endpoints to stay fixed |
| `backbone` | `"InceptionTime"` | Options: `"InceptionTime"`, `"TCN"`, `"CNN"` |
| `lr` | 0.0005 | AdamW with cosine annealing |
| `n_epochs` | 500–1000 | Depends on dataset size |

---

## 5. Loss Functions

| Loss | Flag | Single class? | Notes |
|------|------|:---:|-------|
| **ICAE** | `args.ICAE_loss` | YES | Recommended default |
| WCSS | `args.WCSS_loss` | YES | Add `smoothness_prior` for best results |
| ICAE + Triplet | `args.ICAE_triplets_loss` | NO | Needs >= 2 classes |
| WCSS + Triplet | `args.WCSS_triplets_loss` | NO | Needs >= 2 classes |

---

## 6. Inference

```python
model.eval()
with torch.no_grad():
    X_tensor  = torch.tensor(X_new, dtype=torch.float32).to(device)  # (N, k, signal_len)
    outputs   = model(X_tensor, return_nans=True)
    X_aligned = outputs["data"]        # aligned signals (NaN regions preserved)
    thetas    = outputs["theta"]       # learned warp parameters
    nan_mask  = outputs["nans_mask"]   # mask of padded regions
```

