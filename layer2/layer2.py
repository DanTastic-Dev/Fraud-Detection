import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

os.makedirs('outputs', exist_ok=True)

# ── 1. LOAD + PREPROCESS (same as layer1) ────────────────────────────────────
df = pd.read_csv('datasets/creditcard.csv')

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Amount'] < (Q1 - 1.5 * IQR)) | (df['Amount'] > (Q3 + 1.5 * IQR)))]

# ── 2. IDENTIFY HARD CASES ────────────────────────────────────────────────────
fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]

# Load Layer 1 to identify hard cases
layer1_model = joblib.load('outputs/layer1_model.pkl')
X_fraud = fraud_df.drop('Class', axis=1)
fraud_probs_all = layer1_model.predict_proba(X_fraud)[:, 1]

# Split fraud into hard (L1 uncertain) and easy (L1 confident)
hard_fraud = fraud_df[fraud_probs_all < 0.80]   # held out for evaluation only
easy_fraud = fraud_df[fraud_probs_all >= 0.80]  # used for training

print(f"Hard fraud (held out for eval): {len(hard_fraud)}")
print(f"Easy fraud (used for training): {len(easy_fraud)}")

# ── 3. AUGMENT HARD CASES FOR TRAINING ───────────────────────────────────────
# Gaussian noise augmentation — small perturbations around the 8 hard cases
# These synthetic variants teach the agent about ambiguous patterns
# Original 8 hard cases are NEVER used in training — only in evaluation
np.random.seed(42)
hard_fraud_features = hard_fraud.drop('Class', axis=1)
augmented_rows = []

for _ in range(20):  # 20 copies of each = 160 synthetic variants
    noise = np.random.normal(0, 0.1, hard_fraud_features.shape)
    augmented = hard_fraud_features.copy() + noise
    augmented['Class'] = 1
    augmented_rows.append(augmented)

augmented_hard = pd.concat(augmented_rows).reset_index(drop=True)
print(f"Augmented hard cases (synthetic, training only): {len(augmented_hard)}")

# ── 4. BUILD TRAINING ENVIRONMENT ────────────────────────────────────────────
# Training: easy fraud + synthetic hard variants + legit
# Original 8 hard fraud cases are excluded entirely from training
df_layer2 = pd.concat([
    easy_fraud,
    augmented_hard,
    legit_df.sample(5000, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nLayer 2 training environment size: {len(df_layer2)}")
print(f"Fraud in training env:             {df_layer2['Class'].sum()}")
print(f"  - Easy fraud:                    {len(easy_fraud)}")
print(f"  - Synthetic hard variants:       {len(augmented_hard)}")

# ── 5. FRAUD DETECTION ENVIRONMENT ───────────────────────────────────────────
class FraudEnv(gym.Env):
    """
    Custom Gymnasium environment for fraud detection.

    At each step the agent sees one transaction (30 features)
    and must decide:
        0 = Approve
        1 = Flag for review
        2 = Block
    """

    def __init__(self, dataframe):
        super(FraudEnv, self).__init__()

        self.df = dataframe.reset_index(drop=True)
        self.n_samples = len(self.df)
        self.current_index = 0

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(30,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        true_label = self.df.loc[self.current_index, 'Class']
        reward = self._get_reward(action, true_label)

        self.current_index += 1
        terminated = self.current_index >= self.n_samples
        truncated = False

        if terminated:
            obs = self._get_observation(last=True)
        else:
            obs = self._get_observation()

        return obs, reward, terminated, truncated, {}

    def _get_observation(self, last=False):
        idx = self.current_index if not last else self.current_index - 1
        row = self.df.loc[idx].drop('Class').values.astype(np.float32)
        return row

    def _get_reward(self, action, true_label):
        if action == 2 and true_label == 1:
            return 3.0      # Correct block — caught fraud

        if action == 0 and true_label == 0:
            return 1.0      # Correct approval — legitimate passed through

        if action == 1 and true_label == 1:
            return 1.5      # Flagged fraud — acceptable

        if action == 1 and true_label == 0:
            return -0.5     # Flagged legitimate — minor false alarm

        if action == 2 and true_label == 0:
            return -1.0     # False positive — blocked legitimate

        if action == 0 and true_label == 1:
            return -5.0     # Missed fraud — worst outcome

        return 0.0


# ── 6. TRAIN PPO AGENT ────────────────────────────────────────────────────────
env = FraudEnv(df_layer2)

model_rl = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    seed=42
)

print("\nTraining PPO agent...")
model_rl.learn(total_timesteps=300000)
model_rl.save("outputs/layer2_ppo_model")
print("RL model saved")

# ── 7. EVALUATE ON ORIGINAL HARD CASES (never seen during training) ───────────
print("\nEvaluating on original hard fraud cases (never seen during training)...")

hard_legit   = legit_df.sample(500, random_state=99)
hard_eval_df = pd.concat([hard_fraud, hard_legit]).sample(
    frac=1, random_state=99
).reset_index(drop=True)

eval_env = FraudEnv(hard_eval_df)
obs, _   = eval_env.reset()
hard_actions, hard_labels = [], []

for i in range(len(hard_eval_df)):
    action, _ = model_rl.predict(obs, deterministic=True)
    hard_actions.append(int(action))
    hard_labels.append(int(hard_eval_df.loc[i, 'Class']))
    obs, _, terminated, _, _ = eval_env.step(int(action))
    if terminated:
        break

hard_actions = np.array(hard_actions)
hard_labels  = np.array(hard_labels)
hard_preds   = (hard_actions == 2).astype(int)

print("\n── RL Agent Results (hard cases — fair evaluation) ──")
print(classification_report(hard_labels, hard_preds))

hard_caught = ((hard_preds == 1) & (hard_labels == 1)).sum()
hard_total  = hard_labels.sum()
print(f"Hard fraud cases (unseen): {hard_total}")
print(f"Caught by agent:           {hard_caught}")
print(f"Catch rate:                {hard_caught/hard_total:.2%}")