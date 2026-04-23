import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

os.makedirs('outputs', exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading IEEE-CIS dataset...")

train_t = pd.read_csv('datasets/train_transaction.csv')
train_i = pd.read_csv('datasets/train_identity.csv')

# Merge on TransactionID
df = train_t.merge(train_i, on='TransactionID', how='left')
print(f"Merged shape: {df.shape}")
print(f"Fraud distribution:\n{df['isFraud'].value_counts()}")

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────

# Drop TransactionID — not a feature
df = df.drop(columns=['TransactionID'])

# Drop columns with >50% missing values
thresh = 0.5 * len(df)
df = df.dropna(axis=1, thresh=int(thresh))
print(f"Shape after dropping high-null columns: {df.shape}")

# Separate target
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Label-encode all categorical (object) columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns...")
le = LabelEncoder()
for col in cat_cols:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

# Fill remaining NaNs with median
X = X.fillna(X.median())

# Scale TransactionAmt and TransactionDT
scaler = StandardScaler()
for col in ['TransactionAmt', 'TransactionDT']:
    if col in X.columns:
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Null check: {X.isnull().sum().sum()} nulls remaining")

# Save feature names for layer2/finguard
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'outputs/ieee_feature_names.pkl')
joblib.dump(scaler, 'outputs/ieee_scaler.pkl')
print(f"Feature names saved: {len(feature_names)} features")

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4A. MODEL: Cost-Sensitive (no SMOTE) ──────────────────────────────────────
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {scale:.1f}")

model_cs = XGBClassifier(
    scale_pos_weight=scale,
    n_estimators=100,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist'   # faster on large datasets
)
print("Training Cost-Sensitive model...")
model_cs.fit(X_train, y_train)

# ── 4B. MODEL: SMOTE ──────────────────────────────────────────────────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — fraud: {(y_train_sm==1).sum()}, legit: {(y_train_sm==0).sum()}")

model_sm = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist'
)
print("Training SMOTE model...")
model_sm.fit(X_train_sm, y_train_sm)

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
for name, model in [("Cost-Sensitive", model_cs), ("SMOTE", model_sm)]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n── {name} ──")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# ── 6. SHAP EXPLANATIONS ──────────────────────────────────────────────────────
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(model_cs)
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig('outputs/ieee_shap_summary.png', bbox_inches='tight', dpi=150)
plt.close()
print("SHAP summary plot saved")

# Single fraud explanation
fraud_indices = X_test[model_cs.predict(X_test) == 1].index
if len(fraud_indices) > 0:
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
    plt.savefig('outputs/ieee_shap_single.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Single fraud explanation saved")

# ── 7. DOCUMENT LAYER 1 FAILURES (for Layer 2 routing) ──────────────────────
preds_cs = model_cs.predict(X_test)
proba_cs = model_cs.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['true_label'] = y_test.values
results['predicted']  = preds_cs
results['fraud_prob'] = proba_cs

missed    = results[(results['true_label'] == 1) & (results['predicted'] == 0)]
uncertain = results[(results['fraud_prob'] >= 0.20) & (results['fraud_prob'] <= 0.80)]

print(f"\n── Layer 1 Summary ──")
print(f"Total fraud in test set:    {(y_test == 1).sum()}")
print(f"Caught by Layer 1:          {((preds_cs == 1) & (y_test == 1)).sum()}")
print(f"Missed by Layer 1:          {len(missed)}")
print(f"Uncertain zone (→ Layer 2): {len(uncertain)}")

missed.to_csv('outputs/ieee_layer1_misses.csv', index=False)
uncertain.to_csv('outputs/ieee_layer2_candidates.csv', index=False)

# ── 8. SAVE MODEL ─────────────────────────────────────────────────────────────
joblib.dump(model_cs, 'outputs/ieee_layer1_model.pkl')
print("\nModel saved to outputs/ieee_layer1_model.pkl")