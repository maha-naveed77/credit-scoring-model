import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_and_save():

    # ── STEP 1: Load Dataset ──────────────────────────────────────────
    print("\n STEP 1: Loading German Credit Dataset from OpenML...")
    credit = fetch_openml('credit-g', version=1, as_frame=True)
    df = credit.frame
    print(f"   Loaded {df.shape[0]} applicants with {df.shape[1]} columns")

    # ── STEP 2: Encode Target Column ─────────────────────────────────
    print("\n STEP 2: Encoding target column...")
    print("   'good' → 1  |  'bad' → 0")
    df['class'] = df['class'].map({'good': 1, 'bad': 0})
    print(f"   Distribution: {df['class'].value_counts().to_dict()}")

    # ── STEP 3: Encode Categorical Columns ───────────────────────────
    print("\n STEP 3: Encoding categorical columns with LabelEncoder...")
    cat_cols = df.select_dtypes(include='category').columns.tolist()
    cat_cols.remove('class')

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   {col}: {list(le.classes_)}")

    # ── STEP 4: Feature Engineering ──────────────────────────────────
    print("\n STEP 4: Engineering new features...")
    df['credit_per_month'] = df['credit_amount'] / df['duration']
    df['debt_to_duration'] = df['credit_amount'] / (df['age'] + 1)
    print("   credit_per_month  = credit_amount / duration")
    print("   debt_to_duration  = credit_amount / (age + 1)")

    # ── STEP 5: Split Features and Target ────────────────────────────
    print("\n STEP 5: Splitting features (X) and target (y)...")
    X = df.drop(columns=['class'])
    y = df['class']
    feature_names = X.columns.tolist()
    print(f"   Features: {feature_names}")

    # Save default values (median) for each feature — used in app for unset fields
    defaults = {col: float(X[col].median()) for col in X.columns}

    # ── STEP 6: Train / Test Split ───────────────────────────────────
    print("\n STEP 6: Splitting into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── STEP 7: Scale Features ───────────────────────────────────────
    print("\n STEP 7: Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("   All features now have mean=0, std=1")

    # ── STEP 8: Train Models ─────────────────────────────────────────
    print("\n STEP 8: Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    lr_roc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    print("\n=== Logistic Regression Results ===")
    print(classification_report(y_test, y_pred_lr, target_names=['Bad Credit', 'Good Credit']))
    print(f"ROC-AUC: {lr_roc:.4f}")

    print("\n Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    rf_roc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

    print("\n=== Random Forest Results ===")
    print(classification_report(y_test, y_pred_rf, target_names=['Bad Credit', 'Good Credit']))
    print(f"ROC-AUC: {rf_roc:.4f}")

    # ── STEP 9: Save Everything ──────────────────────────────────────
    print("\n STEP 9: Saving model artifacts to model/ folder...")
    os.makedirs('model', exist_ok=True)

    joblib.dump(rf,            'model/random_forest.pkl')
    joblib.dump(lr,            'model/logistic_regression.pkl')
    joblib.dump(scaler,        'model/scaler.pkl')
    joblib.dump(encoders,      'model/encoders.pkl')
    joblib.dump(feature_names, 'model/feature_names.pkl')
    joblib.dump(defaults,      'model/defaults.pkl')

    print("   model/random_forest.pkl       ← main model used in app")
    print("   model/logistic_regression.pkl ← baseline comparison model")
    print("   model/scaler.pkl              ← same scaler used during training")
    print("   model/encoders.pkl            ← label encoders for categories")
    print("   model/feature_names.pkl       ← ordered list of feature names")
    print("   model/defaults.pkl            ← median values for unset fields")
    print("\n Training complete! Now run:  streamlit run app.py")

if __name__ == '__main__':
    train_and_save()
