# CUP 2026 production pipeline

## Setup

```bash
uv sync
cp .env.example .env
uv run jupyter
```

## Order

1. `train.ipynb` - training pipeline
2. `infer.ipynb` - inference pipeline
