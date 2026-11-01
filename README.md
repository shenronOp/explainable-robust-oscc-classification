# Explainable and Robust Deep Learning for Oral Cancer Histopathology Classification

A deep learning research project focused on classifying oral histopathology images using EfficientNet-B0, with explainability through Grad-CAM and robustness evaluation under image acquisition variations.

## Project Overview

This project investigates the application of deep learning to oral histopathology image classification.

The implemented model classifies histopathology images into five categories:

- MD-OSCC — Moderately Differentiated Oral Squamous Cell Carcinoma
- Normal
- OSMF — Oral Submucous Fibrosis
- PD-OSCC — Poorly Differentiated Oral Squamous Cell Carcinoma
- WD-OSCC — Well Differentiated Oral Squamous Cell Carcinoma

The project explores model performance, explainability and robustness under image variations.

**Note:** This is an academic research project and is not a clinically validated diagnostic system.

## Methodology

The complete experimental methodology is documented separately.

It covers:

- Dataset investigation and duplicate auditing
- Patient-independent data splitting
- EfficientNet-B0 architecture
- Image preprocessing and augmentation
- Model evaluation
- Robustness experiments
- Grad-CAM explainability
- Research limitations

[Read the complete methodology](docs/methodology.md).

## Model Architecture

| Property | Configuration |
|---|---|
| Architecture | EfficientNet-B0 |
| Framework | PyTorch |
| Input resolution | 224 × 224 |
| Number of classes | 5 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Learning approach | Transfer learning |

Selected model checkpoint:

`models/orchid_experiment_best.pth`

## Experimental Results

### Original Test Set

| Metric | Score |
|---|---:|
| Accuracy | 70.01% |
| Macro F1-score | 71.18% |
| Weighted F1-score | 69.64% |

### Mild Color and Brightness Perturbation

| Metric | Score |
|---|---:|
| Accuracy | 66.11% |
| Macro F1-score | 67.03% |
| Weighted F1-score | 65.27% |

The results indicate a performance reduction under the tested image perturbations.

The robustness experiment used one randomized mild brightness, contrast, saturation and hue perturbation per image.

Detailed classification reports, confusion matrices and experiment summaries are available in `reports/`.

## Explainable AI — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize image regions contributing to model class scores.

Implemented functionality:

- Predicted-class Grad-CAM
- Specified-class Grad-CAM
- Single-image visualization
- Batch Grad-CAM generation

Example visualizations are available in `reports/figures/`.

These visualizations are qualitative and do not establish clinical validity.

## Inference

The repository includes a local inference pipeline using the trained EfficientNet-B0 checkpoint.

Supported functionality:

- Single-image classification
- Five-class probability output
- Confidence score
- Batch inference
- CSV export of predictions

## Repository Structure

```text
explainable-robust-oscc-classification/
│
├── configs/
│   ├── orchid_experiment_config.json
│   └── orchid_methodology.json
│
├── data/
│   └── sample/
│
├── docs/
│   └── methodology.md
│
├── models/
│   └── orchid_experiment_best.pth
│
├── notebooks/
│   └── orchid_oscc_classification.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── experiment reports
│
├── src/
│   ├── data/
│   ├── models/
│   │   └── efficientnet.py
│   ├── training/
│   ├── evaluation/
│   │   └── batch_inference.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   └── batch_gradcam.py
│   └── inference/
│       └── predict.py
│
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Dataset: ORCHID OSCC Histopathology Classification Dataset.

The dataset is not included in this repository due to its size. It must be obtained separately from its original source.

The training notebook and methodology documentation describe the dataset preparation and experimental workflow.

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd explainable-robust-oscc-classification
```

Install the required Python dependencies:

```bash
pip install torch torchvision pillow numpy matplotlib
```

Use compatible PyTorch and Torchvision versions for your hardware.

## Running Inference

Run the following command from the repository root:

```bash
python -c "from src.inference.predict import predict_image; print(predict_image('path/to/image.png'))"
```

The output contains:

- Predicted class
- Confidence score
- Probabilities for all five classes

## Running Grad-CAM

Generate an explanation for the predicted class:

```bash
python -c "from src.explainability.gradcam import generate_gradcam; print(generate_gradcam('path/to/image.png', 'reports/figures/gradcam.png'))"
```

Generate an explanation for a specified class:

```bash
python -c "from src.explainability.gradcam import generate_gradcam; print(generate_gradcam('path/to/image.png', 'reports/figures/gradcam_md.png', target_class='MD-OSCC'))"
```

## Running Batch Inference

```bash
python -c "from src.evaluation.batch_inference import run_batch; run_batch('path/to/images', 'reports/metrics/predictions.csv')"
```

## Running Batch Grad-CAM

```bash
python -c "from src.explainability.batch_gradcam import run_batch_gradcam; run_batch_gradcam('path/to/images', 'reports/figures/batch_gradcam')"
```

## Limitations

- Hospital and classification labels are strongly confounded.
- The robustness experiment uses one randomized perturbation per image.
- Grad-CAM visualizations are qualitative.
- The model has not undergone clinical validation.
- The documented experiment uses a different patient split from a subsequent data-preparation experiment. Results must be associated with their respective split and checkpoint.

## Pending Work

The following work can be continued:

- Further research and evaluation.
- Additional explainability experiments.
- Improved robustness and domain generalization studies.
- FastAPI inference backend.
- React and TypeScript frontend.
- Application integration and final documentation.

## Disclaimer

This project is intended for academic and research purposes only. Model predictions must not be used as a substitute for professional medical diagnosis.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
