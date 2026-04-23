import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from stable_baselines3 import PPO

# ── 1. LOAD MODELS ────────────────────────────────────────────────────────────
print("Loading models...")
layer1_model = joblib.load('outputs/layer1_model.pkl')
layer2_model  = PPO.load('outputs/layer2_ppo_model')
print("Both models loaded")

# ── 2. LOAD + PREPROCESS DATA ─────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('datasets/creditcard.csv')

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

# Use a test sample — 10,000 legit + all fraud
fraud_df  = df[df['Class'] == 1]
legit_df  = df[df['Class'] == 0].sample(10000, random_state=99)
test_df   = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=99).reset_index(drop=True)

X_test = test_df.drop('Class', axis=1)
y_test = test_df['Class'].values

print(f"\nTest set: {len(test_df)} transactions ({fraud_df.shape[0]} fraud, 10000 legit)")

# ── 3. ROUTING FUNCTION ───────────────────────────────────────────────────────
def finguard_predict(X, layer1, layer2, low=0.20, high=0.80):
    """
    Routes each transaction through the two-layer system.

    Returns:
        decisions   : final decision per transaction (0=approve, 1=flag, 2=block)
        routed_to   : which layer made the decision ('L1' or 'L2')
    """
    fraud_probs = layer1.predict_proba(X)[:, 1]

    decisions  = np.zeros(len(X), dtype=int)
    routed_to  = np.empty(len(X), dtype=object)

    for i, prob in enumerate(fraud_probs):
        if prob < low:
            # Layer 1 confident: legitimate
            decisions[i] = 0
            routed_to[i] = 'L1'

        elif prob > high:
            # Layer 1 confident: fraud
            decisions[i] = 2
            routed_to[i] = 'L1'

        else:
            # Uncertain — route to Layer 2
            obs = X.iloc[i].values.astype(np.float32)
            action, _ = layer2.predict(obs, deterministic=True)
            decisions[i] = int(action)
            routed_to[i] = 'L2'

    return decisions, routed_to, fraud_probs

# ── 4. RUN FINGUARD ───────────────────────────────────────────────────────────
print("\nRunning FinGuard pipeline...")
decisions, routed_to, fraud_probs = finguard_predict(
    X_test, layer1_model, layer2_model, low=0.01, high=0.80
)
# Add this after the routing call temporarily
l2_mask = routed_to == 'L2'
l2_indices = np.where(l2_mask)[0]

print("\nLayer 2 decision detail:")
for idx in l2_indices:
    obs = X_test.iloc[idx].values.astype(np.float32)
    action, _ = layer2_model.predict(obs, deterministic=True)
    action_name = {0: 'Approve', 1: 'Flag', 2: 'Block'}[int(action)]
    print(f"  Transaction {idx}: prob={fraud_probs[idx]:.4f}, true={y_test[idx]}, action={action_name}")


print("\nFraud probability distribution:")
print(f"  Below 0.20:      {(fraud_probs < 0.20).sum()}")
print(f"  Between 0.20-0.80: {((fraud_probs >= 0.20) & (fraud_probs <= 0.80)).sum()}")
print(f"  Above 0.80:      {(fraud_probs > 0.80).sum()}")
print(f"\nSample of fraud case probabilities:")
fraud_mask = y_test == 1
print(sorted(fraud_probs[fraud_mask])[:20])

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────

# Convert decisions to binary (block=2 or flag=1 → fraud, approve=0 → legit)
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

# Layer 2 performance specifically
l2_mask          = routed_to == 'L2'
l2_preds         = binary_preds[l2_mask]
l2_true          = y_test[l2_mask]
l2_fraud_caught  = ((l2_preds >= 1) & (l2_true == 1)).sum()
l2_fraud_total   = l2_true.sum()

print(f"\n── Layer 2 Specific ──")
print(f"Uncertain transactions routed to L2: {l2_count}")
print(f"Fraud in that group:                 {l2_fraud_total}")
print(f"Caught by Layer 2:                   {l2_fraud_caught}")

# ── 6. SAVE RESULTS ───────────────────────────────────────────────────────────
results_df = X_test.copy()
results_df['true_label'] = y_test
results_df['decision']   = decisions
results_df['routed_to']  = routed_to
results_df['fraud_prob'] = fraud_probs
results_df.to_csv('outputs/finguard_results.csv', index=False)
print("\nFull results saved to outputs/finguard_results.csv")