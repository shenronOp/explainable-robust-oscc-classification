# Methodology

## 1. Dataset

The project uses the ORCHID OSCC Histopathology Classification Dataset.

The original dataset contains 14,705 images across five classification categories:

- MD-OSCC
- Normal
- OSMF
- PD-OSCC
- WD-OSCC

## 2. Data Cleaning

A duplicate audit was conducted using image-content fingerprints.

A total of 306 records belonging to conflicting duplicate groups were excluded, resulting in 14,399 cleaned images.

## 3. Data Splitting

The documented EfficientNet-B0 experiment used a patient-independent split with random seed 42.

| Split | Patients | Images |
|---|---:|---:|
| Training | 100 | 9,557 |
| Validation | 22 | 2,248 |
| Testing | 24 | 2,594 |

## 4. Model Architecture

EfficientNet-B0 was used as the baseline image-classification architecture.

- Framework: PyTorch
- Input resolution: 224 × 224
- Number of output classes: 5

## 5. Preprocessing and Augmentation

**Training transformations:**

- Resize to 224 × 224
- Random horizontal flip
- Random rotation up to 10 degrees
- ImageNet normalization

**Evaluation transformations:**

- Resize to 224 × 224
- Convert to tensor
- ImageNet normalization

## 6. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Confusion matrix

## 7. Initial Robustness Experiment

The baseline model was evaluated under mild brightness, contrast, saturation and hue perturbations.

| Metric | Original | Perturbed |
|---|---:|---:|
| Accuracy | 70.01% | 66.11% |
| Macro-F1 | 71.18% | 67.03% |
| Weighted-F1 | 69.64% | 65.27% |

These results indicate a performance reduction under the tested perturbation.

## 8. Explainability

Grad-CAM was used to generate qualitative visual explanations of model predictions.

These visualizations are exploratory and do not establish clinical validity.

## 9. Limitations

- Hospital and classification labels are strongly confounded.
- The robustness experiment uses one randomized perturbation per image.
- Grad-CAM visualizations do not establish clinical validity.
- The model has not undergone clinical validation.
- The reported experiment uses a different patient split from a subsequent data-preparation experiment; results must be associated with their respective split and checkpoint.