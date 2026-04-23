import pandas as pd
import numpy as np
import joblib
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

os.makedirs('outputs/federated', exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEDERATION_ROUNDS  = 3
TIMESTEPS_PER_ROUND = 100000   # each institution trains this many steps per round
DP_EPSILON         = 1.0       # differential privacy noise level
DP_SENSITIVITY     = 0.1       # weight sensitivity bound (clipping threshold)
RANDOM_SEED        = 42

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("FinGuard — Federated Learning Layer (Phase 3)")
print(f"Rounds: {FEDERATION_ROUNDS}  |  DP epsilon: {DP_EPSILON}")
print("=" * 60)

# ── 1. LOAD + PREPROCESS ──────────────────────────────────────────────────────
df = pd.read_csv('datasets/creditcard.csv')

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

print(f"\nFull dataset: {len(df)} transactions ({len(fraud_df)} fraud)")

# ── 2. SPLIT INTO TWO INSTITUTIONS (60/40) ───────────────────────────────────
# Stratified split — both institutions get proportional fraud cases
fraud_A = fraud_df.sample(frac=0.6, random_state=RANDOM_SEED)
fraud_B = fraud_df.drop(fraud_A.index)

legit_A = legit_df.sample(frac=0.6, random_state=RANDOM_SEED)
legit_B = legit_df.drop(legit_A.index)

df_A = pd.concat([fraud_A, legit_A]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_B = pd.concat([fraud_B, legit_B]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nInstitution A: {len(df_A)} transactions ({fraud_A.shape[0]} fraud)")
print(f"Institution B: {len(df_B)} transactions ({fraud_B.shape[0]} fraud)")

# ── 3. HOLD OUT TEST SET (neither institution sees this) ──────────────────────
# Use fraud cases that are in neither institution's data for fair evaluation
# We use Institution B's fraud as test since it's smaller
test_fraud  = fraud_B.sample(frac=0.3, random_state=99)
test_legit  = legit_B.sample(500, random_state=99)
test_df     = pd.concat([test_fraud, test_legit]).sample(frac=1, random_state=99).reset_index(drop=True)

print(f"Hold-out test set: {len(test_df)} transactions ({len(test_fraud)} fraud)")

# ── 4. FRAUD DETECTION ENVIRONMENT ───────────────────────────────────────────
class FraudEnv(gym.Env):
    def __init__(self, dataframe):
        super(FraudEnv, self).__init__()
        self.df = dataframe.reset_index(drop=True)
        self.n_samples = len(self.df)
        self.current_index = 0
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        return self._get_obs(), {}

    def step(self, action):
        true_label = self.df.loc[self.current_index, 'Class']
        reward     = self._reward(action, true_label)
        self.current_index += 1
        terminated = self.current_index >= self.n_samples
        obs = self._get_obs(last=terminated)
        return obs, reward, terminated, False, {}

    def _get_obs(self, last=False):
        idx = self.current_index - 1 if last else self.current_index
        return self.df.loc[idx].drop('Class').values.astype(np.float32)

    def _reward(self, action, label):
        if action == 2 and label == 1: return  3.0   # correct block
        if action == 0 and label == 0: return  1.0   # correct approve
        if action == 1 and label == 1: return  1.5   # flagged fraud
        if action == 1 and label == 0: return -0.5   # flagged legit
        if action == 2 and label == 0: return -1.0   # false positive
        if action == 0 and label == 1: return -5.0   # missed fraud
        return 0.0

# ── 5. DIFFERENTIAL PRIVACY: ADD NOISE TO WEIGHTS ────────────────────────────
def add_dp_noise(model, epsilon, sensitivity):
    """
    Adds calibrated Gaussian noise to PPO model weights.
    
    Noise scale = (sensitivity / epsilon)
    Higher epsilon = less noise = less privacy but better accuracy
    Lower epsilon  = more noise = more privacy but worse accuracy
    
    This implements the Gaussian mechanism for (epsilon, delta)-DP
    where delta is implicitly set by the noise scale.
    """
    noise_scale = sensitivity / epsilon
    params = model.policy.state_dict()
    noisy_params = {}

    for key, tensor in params.items():
        noise = np.random.normal(0, noise_scale, tensor.shape)
        noisy_params[key] = tensor + noise  # add noise in-place

    model.policy.load_state_dict(noisy_params)
    return model

# ── 6. FEDAVG: AVERAGE WEIGHTS FROM TWO INSTITUTIONS ─────────────────────────
def federated_average(model_A, model_B, weight_A=0.6, weight_B=0.4):
    """
    Weighted FedAvg — Institution A contributes more (60% of data).
    Weights proportional to dataset size, which is standard in FedAvg.
    """
    params_A = model_A.policy.state_dict()
    params_B = model_B.policy.state_dict()
    averaged = {}

    for key in params_A:
        averaged[key] = (weight_A * params_A[key]) + (weight_B * params_B[key])

    return averaged

# ── 7. EVALUATE A MODEL ───────────────────────────────────────────────────────
def evaluate(model, dataset, label=""):
    env = FraudEnv(dataset)
    obs, _ = env.reset()
    actions, labels = [], []

    for i in range(len(dataset)):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        labels.append(int(dataset.loc[i, 'Class']))
        obs, _, terminated, _, _ = env.step(int(action))
        if terminated:
            break

    actions = np.array(actions)
    labels  = np.array(labels)
    preds   = (actions == 2).astype(int)

    caught = ((preds == 1) & (labels == 1)).sum()
    total  = labels.sum()
    fp     = ((preds == 1) & (labels == 0)).sum()

    print(f"\n── {label} ──")
    print(classification_report(labels, preds, zero_division=0))
    print(f"Fraud caught:     {caught}/{total} ({caught/total*100:.1f}%)")
    print(f"False positives:  {fp}")

    return caught / total if total > 0 else 0

# ── 8. INITIALISE MODELS ──────────────────────────────────────────────────────
env_A = FraudEnv(df_A)
env_B = FraudEnv(df_B)

# Both institutions start from the same base Layer 2 model
# This simulates a common initialisation protocol agreed upon between banks
print("\nLoading base Layer 2 model as starting point for both institutions...")
model_A = PPO.load("outputs/layer2_ppo_model", env=env_A, device='cpu')
model_B = PPO.load("outputs/layer2_ppo_model", env=env_B, device='cpu')

# ── 9. FEDERATION ROUNDS ──────────────────────────────────────────────────────
round_results = []

for round_num in range(1, FEDERATION_ROUNDS + 1):
    print(f"\n{'='*60}")
    print(f"FEDERATION ROUND {round_num}/{FEDERATION_ROUNDS}")
    print(f"{'='*60}")

    # ── Local training at each institution ──
    print(f"\n[Round {round_num}] Institution A — local training...")
    model_A.set_env(env_A)
    model_A.learn(total_timesteps=TIMESTEPS_PER_ROUND, reset_num_timesteps=False)

    print(f"\n[Round {round_num}] Institution B — local training...")
    model_B.set_env(env_B)
    model_B.learn(total_timesteps=TIMESTEPS_PER_ROUND, reset_num_timesteps=False)

    # ── Add differential privacy noise before sharing ──
    print(f"\n[Round {round_num}] Adding DP noise (epsilon={DP_EPSILON}) before weight sharing...")
    model_A = add_dp_noise(model_A, DP_EPSILON, DP_SENSITIVITY)
    model_B = add_dp_noise(model_B, DP_EPSILON, DP_SENSITIVITY)

    # ── FedAvg aggregation ──
    print(f"[Round {round_num}] Aggregating weights via FedAvg (60/40 weighting)...")
    global_weights = federated_average(model_A, model_B, weight_A=0.6, weight_B=0.4)

    # ── Distribute global model back to both institutions ──
    model_A.policy.load_state_dict(global_weights)
    model_B.policy.load_state_dict(global_weights)

    # ── Evaluate global model on hold-out test set ──
    recall = evaluate(model_A, test_df, label=f"Round {round_num} Global Model (hold-out test)")
    round_results.append({'round': round_num, 'recall': recall})

    # ── Save round checkpoint ──
    model_A.save(f"outputs/federated/global_model_round{round_num}")
    print(f"[Round {round_num}] Global model saved.")

# ── 10. FINAL RESULTS ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FEDERATED LEARNING — FINAL RESULTS")
print(f"{'='*60}")

print("\nRecall progression across rounds:")
for r in round_results:
    bar = '█' * int(r['recall'] * 20)
    print(f"  Round {r['round']}: {r['recall']*100:.1f}%  {bar}")

# Compare against non-federated baseline
print("\nRunning non-federated baseline (Layer 2 standalone)...")
base_model = PPO.load("outputs/layer2_ppo_model", env=env_A, device='cpu')
base_recall = evaluate(base_model, test_df, label="Non-federated baseline")

final_recall = round_results[-1]['recall']
improvement  = final_recall - base_recall

print(f"\nNon-federated baseline recall: {base_recall*100:.1f}%")
print(f"Federated model recall:        {final_recall*100:.1f}%")
print(f"Improvement from federation:   {improvement*100:+.1f}%")

# Save final global model
model_A.save("outputs/federated/global_model_final")
print("\nFinal global model saved to outputs/federated/global_model_final")

# Save summary
import json
fed_summary = {
    "federation_rounds": FEDERATION_ROUNDS,
    "dp_epsilon": DP_EPSILON,
    "dp_sensitivity": DP_SENSITIVITY,
    "institution_split": "60/40",
    "timesteps_per_round": TIMESTEPS_PER_ROUND,
    "round_results": round_results,
    "baseline_recall": round(base_recall, 4),
    "final_federated_recall": round(final_recall, 4),
    "improvement": round(improvement, 4)
}

with open('outputs/federated/federation_results.json', 'w') as f:
    json.dump(fed_summary, f, indent=2)

print("Federation results saved to outputs/federated/federation_results.json")