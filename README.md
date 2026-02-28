# Heart Disease Classification — Deep Learning with PyTorch, Keras & TensorFlow

> **An end-to-end educational repository** covering the full deep learning workflow — from raw data to explainable predictions — using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).  
> Two notebooks are provided: a focused PyTorch tutorial and a comprehensive multi-framework comparison.

---

## Notebooks

| Notebook | Frameworks | Description |
|----------|-----------|-------------|
| [`heart_disease_deep_learning.ipynb`](heart_disease_deep_learning.ipynb) | PyTorch | 14-step in-depth tutorial — architecture design, regularisation, SHAP explainability |
| [`multi_framework_heart_disease.ipynb`](multi_framework_heart_disease.ipynb) | PyTorch · Keras Sequential · Keras Functional · TF GradientTape | Side-by-side comparison of all major deep learning frameworks |

> **Python version note**: TensorFlow requires Python ≤ 3.12. If you are on Python 3.13+, Parts 2–4 of the multi-framework notebook will be gracefully skipped; PyTorch (Part 1) always runs.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Samples | 303 patients |
| Features | 13 clinical features |
| Target | Binary — `0` No Disease / `1` Disease |
| Missing Values | Yes (6 values, handled via median/mode imputation) |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Integer | Age in years |
| `sex` | Binary | 1 = Male, 0 = Female |
| `cp` | Categorical | Chest pain type (1–4) |
| `trestbps` | Integer | Resting blood pressure (mm Hg) |
| `chol` | Integer | Serum cholesterol (mg/dl) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Integer | Maximum heart rate achieved |
| `exang` | Binary | Exercise-induced angina |
| `oldpeak` | Float | ST depression (exercise vs rest) |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Integer | # major vessels coloured by fluoroscopy (0–3) |
| `thal` | Categorical | 3 = Normal, 6 = Fixed defect, 7 = Reversable defect |

---

## What's Covered

```
Step  1 — Dataset loading via ucimlrepo + target binarisation
Step  2 — Exploratory Data Analysis (EDA)
           - Missing value analysis
           - Class balance check
           - Feature distributions by class
           - Correlation heatmap
           - Categorical feature cross-tabs
Step  3 — Preprocessing
           - Median/mode imputation
           - StandardScaler normalisation
           - Stratified 70 / 15 / 15 train/val/test split
Step  4 — PyTorch Dataset & DataLoader
           - TensorDataset, mini-batching, shuffling
Step  5 — Feedforward Neural Network (FNN)
           - Linear → BatchNorm1d → ReLU → Dropout × N → Linear
Step  6 — Loss Function & Optimizer
           - CrossEntropyLoss (with math derivation)
           - Adam optimizer (momentum + RMSProp)
           - L2 weight decay
Step  7 — Training Loop
           - Forward pass → loss → zero_grad → backward → step
           - train() vs eval() modes explained
Step  8 — Training Curve Visualisation
           - Diagnosing overfitting / underfitting
Step  9 — Evaluation Metrics
           - Accuracy, Precision, Recall, F1, ROC-AUC
           - Confusion Matrix + ROC Curve plots
Step 10 — Regularisation Deep Dive
           - Dropout rate experiment (0.0 → 0.6)
Step 11 — Learning Rate Scheduling
           - StepLR vs ReduceLROnPlateau comparison
           - Live LR tracking plots
Step 12 — Architecture Comparison
           - Logistic Regression vs Shallow vs Medium vs Deep
Step 13 — Feature Importance & Explainability
           - Permutation Feature Importance
           - SHAP DeepExplainer (bar + beeswarm plots)
Step 14 — Summary & Key Takeaways
```

---

## Architecture

```
Input (13 features)
       ↓
Linear(13 → 64) → BatchNorm1d → ReLU → Dropout(0.3)
       ↓
Linear(64 → 32) → BatchNorm1d → ReLU → Dropout(0.3)
       ↓
Linear(32 → 2)  → Logits
```

---

## Key Concepts Explained (with Math)

### ReLU Activation
$$\text{ReLU}(z) = \max(0, z)$$

### Batch Normalisation
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

### Cross-Entropy Loss
$$\mathcal{L} = -\log(\hat{p}_{\text{true class}})$$

### Adam Update Rule
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t$$

### SHAP Shapley Values
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}\,[f(S \cup \{i\}) - f(S)]$$

---

## Requirements

```
torch>=2.0
scikit-learn
ucimlrepo
pandas
numpy
matplotlib
seaborn
shap
```

Install all at once:
```bash
pip install torch scikit-learn ucimlrepo pandas numpy matplotlib seaborn shap
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/donniv86/Educational-Tutorial-PyTorch-Keras-for-Deep-Learning-.git
cd Educational-Tutorial-PyTorch-Keras-for-Deep-Learning-

# (Recommended) create a Python 3.10–3.12 virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Open the PyTorch tutorial
jupyter notebook heart_disease_deep_learning.ipynb

# — or — open the multi-framework notebook
jupyter notebook multi_framework_heart_disease.ipynb
```

Run cells top-to-bottom with `Shift + Enter`. No manual data download needed — the notebooks fetch the dataset directly from UCI.

---

## Results (example)

| Metric | Score |
|--------|-------|
| Test Accuracy | ~0.83 |
| F1 Score | ~0.83 |
| ROC-AUC | ~0.91 |

> Results may vary slightly due to random initialisation.

---

## Top Predictive Features (SHAP & Permutation Importance)

1. `ca` — number of major vessels coloured by fluoroscopy
2. `thal` — thalassemia type
3. `cp` — chest pain type
4. `oldpeak` — ST depression
5. `thalach` — maximum heart rate achieved

These align with known cardiological risk indicators.

---

## Next Steps

- [ ] K-Fold Cross-Validation for more robust evaluation on this small dataset
- [ ] Class-weighted loss for explicit imbalance handling
- [ ] Decision threshold tuning (optimise recall for clinical use)
- [ ] Hyperparameter search with `Optuna`
- [ ] Ensemble of multiple FNN models

---

## Citation

```
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
Heart Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
```

---

## License

This project is for educational purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
