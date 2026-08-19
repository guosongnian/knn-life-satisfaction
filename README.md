# K-Nearest Neighbors: Life Satisfaction

This small educational project implements **k-nearest neighbors (k-NN) regression from scratch** with NumPy and pandas. It predicts a country's life satisfaction from:

- GDP per capita
- Employment rate

The notebooks progress from a one-feature introduction to two-feature modeling, feature scaling, validation, and leave-one-out cross-validation (LOOCV).

The teaching material and notebook explanations are primarily in Japanese.

## Repository structure

```text
.
├── data/
│   └── life_satisfaction_2010_2024.csv
├── notebooks/
│   ├── 04_nearest_neighbors_1_exercises.ipynb
│   ├── 05_nearest_neighbors_2_exercises.ipynb
│   └── 05_nearest_neighbors_2_completed.ipynb
├── src/
│   └── validate_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Notebook sequence

1. **04 — k-NN with one feature (exercises)**  
   Introduces distances, nearest neighbors, regression predictions, prediction error, and selecting `k` using GDP or employment rate separately.

2. **05 — k-NN with two features (exercises)**  
   Adds two-dimensional distance, feature scaling, training/validation/test splits, and LOOCV.

3. **05 — completed version**  
   Contains a complete manual implementation and expected outputs for the second notebook.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Validate the included dataset:

```bash
python src/validate_data.py
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

Then open a notebook under `notebooks/` and run its cells from top to bottom. The notebooks can locate the dataset whether Jupyter starts from the repository root or from the `notebooks/` directory.

## Dataset

`data/life_satisfaction_2010_2024.csv` contains 555 observations covering 2011–2024. The notebooks filter it to 42 countries in 2024.

Required columns:

| Column | Meaning |
|---|---|
| `Country` | Country name |
| `Code` | Three-letter country code |
| `Year` | Observation year |
| `Life Satisfaction` | Regression target |
| `GDP per capita` | Predictor |
| `Employment Rate (%)` | Predictor |

The dataset is included because the notebooks depend on the exact prepared values. Before redistributing the repository publicly, confirm that you have the right to publish the dataset and add its original source/license here.

## Reproduced result

The completed notebook selects `k = 14` using LOOCV. Its final estimate for Japan is:

| Metric | Value |
|---|---:|
| Predicted life satisfaction | 6.656 |
| Observed life satisfaction | 6.147 |
| Absolute error | 0.509 |

## Notes

- The implementation deliberately avoids scikit-learn so that the distance calculation and neighbor selection remain visible.
- Scaling parameters are computed only from the active training fold, avoiding validation-data leakage.
- Japan is held out as the final test observation and is not used to choose `k`.

