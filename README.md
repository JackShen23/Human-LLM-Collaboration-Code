# Human–LLM Collaboration: Figures and PKHMM Code

This repository contains code for reproducing the main figures and fitting the
covariate-dependent hidden Markov model (PKHMM) used in our study of human–LLM
collaboration under time pressure.

## Repository structure

```text
├── _PLOT/                 # Figure scripts and example outputs
├── _PKHMM-MODEL/         # PKHMM training and model documentation
├── _DATA/README.md        # Data layout and variable dictionary
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data access and setup

The participant-level data are distributed separately through Figshare. The
record remains private during peer review and will be made public after article
publication. Extract the downloaded archive and either place `_DATA` in the
repository root or set its location explicitly:

```bash
export HLC_DATA_DIR=/absolute/path/to/_DATA
```

On Windows PowerShell:

```powershell
$env:HLC_DATA_DIR = "C:\path\to\_DATA"
```

The required files and variables are described in
[`_DATA/README.md`](_DATA/README.md). Do not commit participant data to the
public GitHub repository.

## Workflows

Generate all included figures:

```bash
python _PLOT/run_all.py
```

Run individual figure groups:

```bash
python _PLOT/plot_human_ai_2.py
python _PLOT/plot_accuracy_by_round.py
```

Fit the PKHMM candidates:

```bash
python _PKHMM-MODEL/train_pkhmm.py
```

## Analysis notes

- Each participant file contains up to 60 collaboration rounds; missing rounds
  are not imputed.
- `round_score == 0` denotes a correct final decision in the figure scripts.
- PKHMM decision types 0, 1, and 2 represent rejection, acceptance, and timeout;
  timeouts follow the prespecified rule and are treated as acceptance.
- Model constants and random seeds are defined in `train_pkhmm.py`.

## License and citation

Code is released under the [MIT License](LICENSE). The data are governed by the
terms and license attached to the Figshare record. Cite the article and software
release using [CITATION.cff](CITATION.cff); We will update provisional article
metadata when the final DOI is available.
