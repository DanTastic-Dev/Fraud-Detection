import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('datasets/creditcard.csv')
print(df.shape)
print(df['Class'].value_counts())

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
# 'Amount' and 'Time' are the only non-PCA columns, scale them
print(df.isnull().sum())  
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

X = df.drop('Class', axis=1)
y = df['Class']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3A. MODEL: Cost-Sensitive (no SMOTE) ──────────────────────────────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()  # ~577

model_cs = XGBClassifier(
    scale_pos_weight=scale,
    n_estimators=100,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
model_cs.fit(X_train, y_train)

# ── 3B. MODEL: SMOTE version ───────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model_sm = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
model_sm.fit(X_train_sm, y_train_sm)

# ── 4. EVALUATE ───────────────────────────────────────────────────────────────
for name, model in [("Cost-Sensitive", model_cs), ("SMOTE", model_sm)]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n── {name} ──")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# ── 5. SHAP EXPLANATIONS ──────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model_cs)
# Only run on a sample — full dataset will be slow
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig('outputs/shap_summary.png', bbox_inches='tight', dpi=150)
plt.close()
print("SHAP plot saved to shap_summary.png")

# Pick one flagged transaction and explain it
fraud_indices = X_test[model_cs.predict(X_test) == 1].index
one_fraud = X_test.loc[[fraud_indices[0]]]

shap.waterfall_plot(
    shap.Explanation(
        values=explainer.shap_values(one_fraud)[0],
        base_values=explainer.expected_value,
        data=one_fraud.iloc[0],
        feature_names=X_test.columns.tolist()
    ),
    show=False
)
plt.savefig('outputs/shap_single_explanation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Single explanation saved")

# ── 6. DOCUMENT FAILURES (justification for Layer 2) ─────────────────────────
preds_cs = model_cs.predict(X_test)
proba_cs = model_cs.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['true_label']    = y_test.values
results['predicted']     = preds_cs
results['fraud_prob']    = proba_cs

# Missed frauds — Layer 2 needs to catch these
missed = results[(results['true_label'] == 1) & (results['predicted'] == 0)]

# Uncertain zone — route these to Layer 2
uncertain = results[(results['fraud_prob'] >= 0.20) & (results['fraud_prob'] <= 0.80)]

print(f"Total fraud in test set:   {(y_test == 1).sum()}")
print(f"Caught by Layer 1:         {((preds_cs == 1) & (y_test == 1)).sum()}")
print(f"Missed by Layer 1:         {len(missed)}")
print(f"Uncertain zone (→ Layer 2): {len(uncertain)}")

missed.to_csv('outputs/layer1_misses.csv', index=False)
uncertain.to_csv('outputs/layer2_candidates.csv', index=False)

print("\nMissed fraud probability scores:")
print(missed['fraud_prob'].values)
joblib.dump(model_cs, 'outputs/layer1_model.pkl')
print("Model saved")