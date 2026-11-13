# PKHMM model training

`train_pkhmm.py` fits the covariate-dependent hidden Markov model separately to
the `Human-AI (low)`, `Human-AI (medium)`, and `Human-AI (high)` conditions.

The model includes:

1. an initial-state model using anonymized participant categories and the first
   scenario features;
2. non-homogeneous state transitions using current scenario features and
   previous-round feedback;
3. state-dependent AI acceptance using the same time-varying covariates.

Run from the repository root:

```bash
python _PKHMM-MODEL/train_pkhmm.py
```

The script reads only files under `_DATA`. 
Results are written to `_PKHMM-MODEL/results/`.

