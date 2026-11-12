<div align="center">
  <img src="assets/banner.svg" alt="Credit Card Fraud Detection — preprocessing and Random Forest" width="100%">
  <br><br>
  <h2>0.17% of transactions. One model. Zero leakage.</h2>
  <p>
    A deliberately small fraud detection pipeline: robust preprocessing and a
    class-balanced Random Forest, evaluated on a held-out split.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/model-Random%20Forest-22C55E?style=flat-square" alt="Random Forest">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/executed-end--to--end-22C55E?style=flat-square" alt="Executed end to end">
  </p>
  <p>
    <a href="credit_card_fraud_detection.ipynb"><img src="https://img.shields.io/badge/Open-Notebook-0F172A?style=for-the-badge&logo=jupyter&logoColor=white" alt="Open notebook"></a>
    <a href="https://colab.research.google.com/github/aryanmendpara/credit-card-fraud-detection/blob/main/credit_card_fraud_detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"><img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle dataset"></a>
  </p>
</div>

---

> **The point:** predicting “legitimate” for everything scores 99.83% accuracy and catches zero fraud.
>
> This pipeline keeps only what matters — **preprocessing and the model** — and evaluates it with metrics that actually reflect the fraud class.

## The data

The public Kaggle dataset contains **284,807** European card transactions and **492** frauds — a **0.1727%** positive rate. Features `V1`–`V28` are PCA components; `Time` and `Amount` are raw.

| Transactions | Fraud cases | Prevalence | Imbalance |
|---:|---:|---:|---:|
| 284,807 | 492 | 0.1727% | ≈ 578 : 1 |

The notebook requires `creditcard.csv` in the project root. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) before running.

## The pipeline

```mermaid
flowchart TD
    A["creditcard.csv<br/>284,807 × 31"] --> B["Stratified 80 / 20 split"]
    B -->|"train"| C["ColumnTransformer<br/>RobustScaler(Time, Amount)<br/>V1–V28 passthrough"]
    C --> D["RandomForestClassifier<br/>100 trees · balanced_subsample"]
    B -->|"test — untouched"| E["Held-out evaluation"]
    D --> E
    E --> F["PR-AUC · ROC-AUC<br/>Precision · Recall · F1<br/>Confusion matrix"]

    classDef data fill:#0b1220,stroke:#22d3ee,color:#e2e8f0,stroke-width:1.5px
    classDef model fill:#052e1a,stroke:#34d399,color:#dcfce7,stroke-width:1.5px
    classDef eval fill:#0b1220,stroke:#334155,color:#e2e8f0

    class A,B data
    class C,D model
    class E,F eval
```

That is the whole architecture:

```python
preprocessor = ColumnTransformer(
    transformers=[("robust", RobustScaler(), ["Time", "Amount"])],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
```

- **`RobustScaler` only on `Time` and `Amount`.** Median/IQR scaling is resistant to the extreme transaction values; `V1`–`V28` are already PCA-transformed.
- **`class_weight="balanced_subsample"`.** Each bootstrap sample is re-weighted so the rare fraud class is not ignored.
- **Everything is inside one `Pipeline`.** The scaler is fitted on the training fold only; the test set is never transformed during fitting.
- **No SMOTE, no model zoo, no threshold tuning.** The class weighting handles imbalance inside the model.

## Results

| Model | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.8472** | **0.9580** | **0.9615** | **0.7653** | **0.8523** |

At the default `0.5` cutoff, the model catches **75 of 98 frauds** with **3 false alerts** on the 56,962-transaction test split.

<p align="center">
  <img src="assets/model-evaluation.png" alt="Random Forest precision-recall curve and confusion matrix" width="900">
  <br>
  <sub>Left: PR curve against the 0.17% no-skill baseline. Right: confusion matrix at the default threshold.</sub>
</p>

Accuracy is deliberately omitted: at this prevalence, a model that never predicts fraud would be 99.83% “accurate”.

## Quickstart

```bash
git clone https://github.com/aryanmendpara/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab credit_card_fraud_detection.ipynb
```

Then choose **Kernel → Restart & Run All**.

Download [`creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root to use the real data. The 150 MB CSV is intentionally gitignored.

## Project structure

```
.
├── credit_card_fraud_detection.ipynb   # preprocessing + model + evaluation
├── assets/
│   ├── banner.svg
│   └── model-evaluation.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Design notes

- **Split before fitting.** Stratification preserves the fraud rate and keeps the test set out of every fit call.
- **One model.** The Random Forest is the selected model; no baseline comparison is retained.
- **PR-AUC over accuracy.** It measures ranking quality on the fraud class instead of rewarding the majority class.
- **Default `0.5` threshold.** If precision/recall needs to shift later, tune the threshold against real cost assumptions — not in this minimal baseline.

**Limitations:** the split is random rather than time-based, and the threshold is the default.

---

<div align="center">
  <p>
    <strong>Built with</strong> Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
  </p>
  <p>
    Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud Detection</a> by the Machine Learning Group, Université Libre de Bruxelles.
  </p>
</div>
