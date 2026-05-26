# Beetle Taxonomic Classification (Cicindela & Platydracus) using Deep Learning

This repository contains code for taxonomic classification of tiger beetles (*Cicindela*) and rove beetles (*Platydracus*) using a single deep-learning multilabel classifier built with fastai and PyTorch.

## Citation

This code was created by Bruno A. S. de Medeiros and is used in the following paper:

Postema, E. G., Briscoe, L., Harder, C., Hancock, G. R. A., Guarnieri, L. D., Eisel, T., Welch, K., Fischer, N., Johnson, C., de Souza, D., Phillip, D., Baquiran, R., Sepulveda, T., & de Medeiros, B. A. S. (2025). DrawerDissect: Whole-drawer insect imaging, segmentation, and transcription using AI. EcoEvoRxiv. https://doi.org/10.32942/X2QW84

## Overview

One multi-label classifier is trained on the union of *Cicindela* and *Platydracus* specimen images. The label vocabulary is a four-level hierarchy:

- **Genus**: capitalized — `Cicindela`, `Platydracus`.
- **Species group** (`<genus>group_<group>`): morphologically diagnosable groups within a genus, e.g. `platydracusgroup_femoratus`. Currently defined only for *Platydracus* (14 groups) via `species_groups.csv`; *Cicindela* specimens carry no group label.
- **Species**: genus-prefixed, lowercase — e.g. `cicindela_repanda`, `platydracus_angusticeps`.
- **Subspecies** (when present): genus-prefixed with two or more underscores — e.g. `cicindela_repanda_repanda`.

Genus-prefixing guarantees uniqueness across taxa and makes the taxonomic level of every label decidable: capitalized → genus; contains `group_` → species group; one underscore → species; two or more underscores → subspecies. At evaluation time the test set is split by genus so per-genus precision/recall can be reported alongside global metrics.

## Data layout

Images are expected under `images/`, which is a symlink to the drive-mounted data root:

```
images -> /data/bdemedeiros/DrawerDissect_Taxon_ID_data/
  Cicindela/{train,valid,test}/<species_[subsp]>/*.png
  Platydracus/{train,valid,test}/<species_[subsp]>/*.png
```

Any additional genus can be added later by dropping another directory under the data root with the same `{train,valid,test}/<species_[subsp]>/` structure — the notebook discovers genera at runtime via `GENERA = sorted(…images_root.iterdir()…)`.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fastai
```

## Usage

### Training

```bash
jupyter lab Run_multilabel_fastai.ipynb
```

The notebook performs single-label pretraining, transfers weights into a multi-label head, fine-tunes on the combined dataset, evaluates on the held-out test set, and writes `test_results_<arch>.csv` with `Term`, `genus`, `taxonomic_level`, and per-class metrics. A per-genus and per-(genus, level) micro-averaged summary is also printed.

### Performance analysis

```bash
Rscript -e "rmarkdown::render('plot_precision_recall.Rmd')"
```

Produces per-genus faceted precision/recall, training-size, and F1 figures (`taxonomy_metrics.pdf` / `.png`).

### HuggingFace upload

```bash
python push_to_hf.py --dry-run    # inspect config without uploading
python push_to_hf.py              # upload to the default repo
```

## Key Features

- **Four-level hierarchy** (genus / species group / species / subspecies), predicted jointly.
- **Weighted sampling**: square-root inverse-frequency weighting addresses class imbalance.
- **Mixed precision, multi-GPU** (configured for 2 GPUs).
- **Modern vision backbone**: EVA-02 Large / ResNeXt options.

## Model Files

Trained models are saved in `exported_fastai_models/`:
- `*_sl.pkl`: single-label pretrained models
- `*_ml.pkl`: final multi-label models

Trained model weights are not part of this repository due to their large size. See https://huggingface.co/brunoasm/Cicindela_Platydracus_ID_FMNH for the final multilabel model.

## Hardware Requirements

- Multi-GPU setup (configured for 2 GPUs via `device_ids=[0,1]`).
- Substantial GPU memory for large batch processing.
- Mixed-precision training support.

## Notes

- Hybrid-specimen folders (`*_x_*`) were canonicalized on disk so the same folder name appears in train, valid, and test splits.

## File Contents

### Core Files
- `Run_multilabel_fastai.ipynb` - Main training pipeline and model development notebook
- `environment.yml` - Conda environment specification with all dependencies
- `push_to_hf.py` - Script to upload trained models to Hugging Face Hub
- `CLAUDE.md` - Development guidelines and project documentation for AI assistants

### Data and Images
- `images/` - Dataset directory containing train/valid/test splits organized by taxonomic labels. The actual files are not available in this repository due to storage size.
- `unknowns/` - Test images for model evaluation and inference

### Models and Results
- `exported_fastai_models/` - Directory containing trained model files (.pkl format)
- `models/` - Additional model storage and configurations
- `test_results_*.csv` - Model performance metrics by taxonomic classification
- `taxonomy_metrics.pdf` and `taxonomy_metrics.png` - Performance visualization charts

### Analysis and Visualization
- `plot_precision_recall.Rmd` - R Markdown script for generating performance analysis plots
- `plot_precision_recall.nb.html` - Rendered HTML output of the precision-recall analysis

### Project Configuration
- `Beetle_taxon_id.Rproj` - R Studio project configuration file
- `LICENSE` - Project license information
- `README.md` - This documentation file

## License

See LICENSE file for details.
