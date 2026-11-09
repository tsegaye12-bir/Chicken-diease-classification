# Chicken disease classification reproducibility package

This package contains the article, source code, model checkpoints, manifests, metrics, figures, and XAI outputs for the 30-epoch experiment described in the manuscript.

## Main files

- `chicken_disease_classification_article.md`: revised article with separate model-based Results subsections.
- `run_study.py`: downloads are not performed by this script; it expects the Kaggle dataset at the path documented in the script and trains CustomCNN, ResNet18, MobileNetV3-Small, and EfficientNet-B0 for 30 epochs. The package also includes the separately trained regularized CNN–Transformer experiment in `run_cnn_transformer.py`.
- `generate_individual_figures.py`: creates separate model-specific accuracy/loss figures, classification-report heat maps, confusion matrices, and class-wise metric figures.
- `outputs/`: model checkpoints, manifests, metrics, combined figures, individual model figures, and XAI comparison.

## Reproduction notes

The experiment used random seed 42, a stratified 70/15/15 split, 128×128 model inputs, batch size 64, AdamW, cosine-annealing learning rate, and training-only horizontal flip, ±20° rotation, and color jitter. The original Kaggle images are not redistributed in this package. Download the dataset from the URL in `outputs/metadata.json` and update `ROOT` in `run_study.py` if the local path differs.

The XAI figure is generated after training and uses the best ResNet18 checkpoint. It shows Grad-CAM, Grad-CAM++, and integrated-gradient-style attribution on one correctly classified example per class.
