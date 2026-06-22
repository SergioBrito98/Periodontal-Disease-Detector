# Periodontal Disease Classification Using EfficientNet-B2

Deep learning-based prototype for automated classification of periodontal conditions (Healthy, Gingivitis, Periodontitis) from intraoral RGB photographs.

## Overview

This repository contains the training and evaluation code for the study:

> **Artificial Intelligence-Based Prototype for Early Diagnosis of Gingivitis and Periodontitis in Adults**
>
> Pintado-Brito S.D., Izquierdo-Vega J.A., Ortega-Palacios R., Izquierdo-Vega A.J., Soria-Alcaraz J.A., Salgado-Ramirez C.D.
>
> Published in *BioMedInformatics* (MDPI)

The model uses **EfficientNet-B2** with a two-phase transfer learning strategy to classify intraoral photographs into three periodontal categories. It was trained on a combined dataset of 1,552 images from two sources: a proprietary clinical dataset (MIO, 765 images) and a publicly available Mendeley repository (787 images).

## Architecture

- **Backbone**: EfficientNet-B2 (ImageNet pre-trained)
- **Classification head**: BN - Dropout(0.4) - Linear(1408, 512) - SiLU - BN - Dropout(0.2) - Linear(512, 3)
- **Training**: Two-phase (Phase 1: frozen backbone, 25 epochs; Phase 2: full fine-tuning, 50 epochs)
- **Loss**: Focal Loss (gamma=1.5) with label smoothing (0.1)
- **Augmentation**: CutMix (p=0.5) / Mixup (alpha=0.3) + Albumentations pipeline
- **Explainability**: Grad-CAM++

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support

### Installation

```bash
pip install -r requirements.txt
```

## Dataset

The model was trained on:

| Source | Healthy | Gingivitis | Periodontitis | Total |
|--------|---------|------------|---------------|-------|
| MIO (proprietary) | 265 | 255 | 245 | 765 |
| Mendeley (public) | 81 | 706 | — | 787 |
| **Total** | **346** | **961** | **245** | **1,552** |

- **MIO dataset**: Clinical photographs captured between March–September 2024 using a Canon EOS R50 camera. Available at [Zenodo DOI pending].
- **Mendeley dataset**: [Dental Conditions Dataset](https://data.mendeley.com/)

## Usage

### Training

```bash
python train.py
```

Key configuration parameters are defined in the `CFG` dictionary at the top of `train.py`. Modify paths and hyperparameters as needed:

```python
CFG = {
    "img_size": 260,
    "batch_size": 16,
    "lr_phase1": 3e-4,
    "lr_phase2": 5e-5,
    "epochs_phase1": 25,
    "epochs_phase2": 50,
    "focal_gamma": 1.5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.3,
    "cutmix_prob": 0.5,
    "num_classes": 3,
}
```

### Data Augmentation Hyperparameters

The training pipeline uses the following augmentation parameters (Albumentations library):

| Augmentation | Parameters | Probability |
|-------------|------------|-------------|
| HorizontalFlip | — | 0.5 |
| VerticalFlip | — | 0.25 |
| ColorJitter | brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08 | 0.7 |
| RandomBrightnessContrast | brightness_limit=0.25, contrast_limit=0.25 | 0.5 |
| RandomGamma | gamma_limit=(70, 130) | 0.4 |
| ImageCompression | quality_lower=35, quality_upper=80 | 0.5 |
| GaussianBlur | blur_limit=3 | 0.2 |
| CLAHE | clip_limit=2.0, tile_grid_size=(8, 8) | 0.2 |

Additionally, Mixup (alpha=0.3) and CutMix (p=0.5, alpha=1.0) are applied at the batch level.

## Preprocessing

All images undergo:
1. Vertical anatomical cropping (20%–80% of image height) to focus on the dentogingival region
2. Resizing to 260 x 260 pixels
3. Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.833 | 0.789–0.874 |
| F1 Macro | 0.832 | — |
| AUC Macro | 0.962 | 0.946–0.976 |
| MCC | 0.718 | — |
| Cohen's Kappa | 0.705 | — |

## License

This project is licensed under the MIT License.

## AI Assistance Disclosure

Portions of this code were developed with the assistance of AI-based tools (Claude, Anthropic) for code optimization and debugging. All outputs were reviewed, validated, and adapted by the authors.

## Citation

If you use this code, please cite:

```bibtex
@article{pintado2025periodontal,
  title={Artificial Intelligence-Based Prototype for Early Diagnosis of Gingivitis and Periodontitis in Adults},
  author={Pintado-Brito, Sergio David and Izquierdo-Vega, Jeannet Alejandra and Ortega-Palacios, Roc{\'i}o and Izquierdo-Vega, Aleli Julieta and Soria-Alcaraz, Jorge A. and Salgado-Ramirez, Carlos Daniel},
  journal={BioMedInformatics},
  year={2025},
  publisher={MDPI}
}
```
