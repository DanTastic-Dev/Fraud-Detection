import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from stable_baselines3 import PPO

# ── 1. LOAD MODELS ────────────────────────────────────────────────────────────
print("Loading models...")
layer1_model = joblib.load('outputs/ieee_layer1_model.pkl')
layer2_model  = PPO.load('outputs/ieee_layer2_ppo_model')
print("Both models loaded successfully")

# ── 2. LOAD + PREPROCESS DATA (mirrors layer1/layer2 exactly) ────────────────
print("\nLoading IEEE-CIS dataset...")

train_t = pd.read_csv('datasets/train_transaction.csv')
train_i = pd.read_csv('datasets/train_identity.csv')

df = train_t.merge(train_i, on='TransactionID', how='left')
df = df.drop(columns=['TransactionID'])

thresh = 0.5 * len(df)
df = df.dropna(axis=1, thresh=int(thresh))

X_full = df.drop(columns=['isFraud'])
y_full = df['isFraud']

cat_cols = X_full.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    X_full[col] = X_full[col].astype(str)
    X_full[col] = le.fit_transform(X_full[col])

X_full = X_full.fillna(X_full.median())

scaler = StandardScaler()
for col in ['TransactionAmt', 'TransactionDT']:
    if col in X_full.columns:
        X_full[col] = scaler.fit_transform(X_full[col].values.reshape(-1, 1))

# ── 3. BUILD TEST SET (all fraud + 10,000 legit) ─────────────────────────────
df_processed = X_full.copy()
df_processed['isFraud'] = y_full.values

fraud_df = df_processed[df_processed['isFraud'] == 1]
legit_df  = df_processed[df_processed['isFraud'] == 0].sample(10000, random_state=99)

test_df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=99).reset_index(drop=True)
X_test  = test_df.drop(columns=['isFraud'])
y_test  = test_df['isFraud'].values

print(f"Test set: {len(test_df)} transactions ({len(fraud_df)} fraud, 10000 legit)")

# ── 4. ROUTING FUNCTION ───────────────────────────────────────────────────────
def finguard_predict(X, layer1, layer2, low=0.20, high=0.80):
    """
    Routes each transaction through the two-layer FinGuard system.

    Layer 1 (XGBoost):
        - prob < low  → Approve (confident legit)
        - prob > high → Block   (confident fraud)
        - else        → Route to Layer 2

    Layer 2 (PPO RL Agent):
        - 0 = Approve
        - 1 = Flag for review
        - 2 = Block

    Returns:
        decisions  : final decision per transaction
        routed_to  : 'L1' or 'L2'
        fraud_probs: Layer 1 probability scores
    """
    fraud_probs = layer1.predict_proba(X)[:, 1]

    decisions = np.zeros(len(X), dtype=int)
    routed_to = np.empty(len(X), dtype=object)

    for i, prob in enumerate(fraud_probs):
        if prob < low:
            decisions[i] = 0
            routed_to[i] = 'L1'
        elif prob > high:
            decisions[i] = 2
            routed_to[i] = 'L1'
        else:
            obs = X.iloc[i].values.astype(np.float32)
            action, _ = layer2.predict(obs, deterministic=True)
            decisions[i] = int(action)
            routed_to[i] = 'L2'

    return decisions, routed_to, fraud_probs

# ── 5. RUN FINGUARD ───────────────────────────────────────────────────────────
print("\nRunning FinGuard pipeline...")
decisions, routed_to, fraud_probs = finguard_predict(
    X_test, layer1_model, layer2_model, low=0.20, high=0.80
)

# ── 6. FRAUD PROBABILITY DISTRIBUTION ────────────────────────────────────────
print("\nFraud probability distribution:")
print(f"  Below 0.20:        {(fraud_probs < 0.20).sum()}")
print(f"  Between 0.20-0.80: {((fraud_probs >= 0.20) & (fraud_probs <= 0.80)).sum()}")
print(f"  Above 0.80:        {(fraud_probs > 0.80).sum()}")

# Layer 2 detail
l2_mask    = routed_to == 'L2'
l2_indices = np.where(l2_mask)[0]
print(f"\nLayer 2 routed {len(l2_indices)} transactions. Sample decisions:")
for idx in l2_indices[:10]:
    action_name = {0: 'Approve', 1: 'Flag', 2: 'Block'}[decisions[idx]]
    print(f"  Transaction {idx}: prob={fraud_probs[idx]:.4f}, true={y_test[idx]}, action={action_name}")

# ── 7. EVALUATE ───────────────────────────────────────────────────────────────
binary_preds = (decisions >= 1).astype(int)

print("\n── FinGuard Combined Results ──")
print(classification_report(y_test, binary_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, fraud_probs):.4f}")

# Routing breakdown
l1_count = (routed_to == 'L1').sum()
l2_count = (routed_to == 'L2').sum()
print(f"\n── Routing Breakdown ──")
print(f"Handled by Layer 1:  {l1_count} ({l1_count/len(test_df)*100:.1f}%)")
print(f"Routed to Layer 2:   {l2_count} ({l2_count/len(test_df)*100:.1f}%)")

# Layer 2 performance
l2_preds        = binary_preds[l2_mask]
l2_true         = y_test[l2_mask]
l2_fraud_caught = ((l2_preds >= 1) & (l2_true == 1)).sum()
l2_fraud_total  = l2_true.sum()

print(f"\n── Layer 2 Specific ──")
print(f"Uncertain transactions routed to L2: {l2_count}")
print(f"Fraud in that group:                 {l2_fraud_total}")
print(f"Caught by Layer 2:                   {l2_fraud_caught}")
if l2_fraud_total > 0:
    print(f"Layer 2 catch rate:                  {l2_fraud_caught/l2_fraud_total:.2%}")

# ── 8. SAVE RESULTS ───────────────────────────────────────────────────────────
results_df = X_test.copy()
results_df['true_label'] = y_test
results_df['decision']   = decisions
results_df['routed_to']  = routed_to
results_df['fraud_prob'] = fraud_probs
results_df.to_csv('outputs/ieee_finguard_results.csv', index=False)
print("\nFull results saved to outputs/ieee_finguard_results.csv")