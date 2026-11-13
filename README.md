# Generative AI Applications: Conditional DCGAN on Fashion-MNIST

## Overview

This project trains a conditional DCGAN that generates `28x28` grayscale Fashion-MNIST garment images, conditioned on one of the `10` class labels. An evaluation classifier trained only on real images reads the generated batches: macro class agreement `0.9008` against a `0.10` chance level, with per-class agreement, feature-space Fréchet distances (`20.9619` generated vs `1.3127` for the real-vs-real reference), intra-class diversity ratios, and a failure analysis all in the notebook.

## Repository

- `generative_model.ipynb`: executed notebook with data loading, checksum verification, model definition, training, evaluation, figures, and a closing summary
- `Generative_AI_Analysis_Report.md` and `Generative_AI_Analysis_Report.pdf`: the written report and the print-ready report
- `figures/`: the seven report figures, exported by the notebook
- `DATASET.md`: dataset access, contents, checksums, license
- `requirements.txt`: pinned package versions
- `build/build_report.py`: the gated builder that renders the report PDF from the committed Markdown
- `data/`: the dataset lands here on first run (see DATASET.md)

## Reproduce

`requirements.txt` was generated with `pip freeze`, so install its pinned packages with `pip install -r requirements.txt` in an activated Python `3.13` environment.
The `uv` commands below show the same setup with `uv`; the CUDA wheel index is required because `torch==2.11.0+cu128` is not on PyPI:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r requirements.txt
python -m nbconvert --to notebook --execute --inplace generative_model.ipynb
```

The first execution downloads Fashion-MNIST (four `.gz` archives, about `31` MB) under `data/FashionMNIST/raw/` and is offline-safe afterward; the notebook verifies the archive checksums before training.
If you regenerate `requirements.txt`, avoid PowerShell `>` because it writes UTF-16.
A CPU-only machine runs the same code, only slower.

## Dataset

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) from Zalando Research
- `70,000` grayscale product photographs at `28x28` pixels, `10` classes with `7,000` images each; `60,000` train / `10,000` test
- License: MIT (the upstream Fashion-MNIST repository); download checksums recorded in `DATASET.md`

## License

Shared for portfolio and review purposes only, no reuse license is granted, and no license file is included.
The dataset is public data under its own terms (MIT); see `DATASET.md`.
