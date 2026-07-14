# Credit Scoring Model

A machine learning web app that predicts credit risk (Good/Bad) for bank loan applicants using a Random Forest classifier trained on the German Credit Dataset.

---

## Project Structure

```
credit-scoring/
├── train.py           
├── app.py             
├── requirements.txt   
├── README.md          
└── model/             
    ├── random_forest.pkl
    ├── logistic_regression.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── feature_names.pkl
    └── defaults.pkl
```

---

## How to Run

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Train the model:**
```bash
python train.py
```
This downloads the dataset, trains both models, evaluates them, and saves all artifacts to the `model/` folder.

**Step 3 — Launch the app:**
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`

---

## ML Pipeline

| Step | What Happens |
|------|-------------|
| Data | German Credit Dataset — 1000 applicants, 20 features (via sklearn/OpenML) |
| Encoding | 13 categorical features label-encoded |
| Feature Engineering | 2 derived features: `credit_per_month`, `debt_to_duration` |
| Scaling | StandardScaler (mean=0, std=1) |
| Models | Logistic Regression (baseline) + Random Forest (100 trees) |
| Evaluation | Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix |

---

## Results

| Model | Accuracy | ROC-AUC | Bad Credit F1 |
|-------|----------|---------|---------------|
| Logistic Regression | 69% | 0.73 | 0.42 |
| **Random Forest** | **77%** | **0.79** | **0.52** |

**Top predictive features:**
1. `credit_per_month` (engineered) — monthly loan burden
2. `debt_to_duration` (engineered) — loan relative to applicant age
3. `credit_amount` — total loan size
4. `checking_status` — current account balance
5. `duration` — loan repayment period

---

##  Key Observations

- **Class imbalance** (700 good vs 300 bad) causes lower recall on bad credit cases — the model sees far more good examples during training
- **Feature engineering added value** — the two derived features ranked #1 and #2 in importance, outperforming all original columns
- **Random Forest vs Logistic Regression** — RF captures non-linear feature interactions that LR's linear boundary cannot

---

##  Tech Stack

- Python 3.10+
- scikit-learn — model training and evaluation
- pandas / numpy — data processing
- Streamlit — web interface
- matplotlib — visualizations
- joblib — model serialization

---

##  Dataset

**German Credit Dataset** via OpenML (`credit-g`, version 1)  
1000 applicants · 20 features · Binary classification (good/bad credit)  
Source: Professor Dr. Hans Hofmann, University of Hamburg (1994)
