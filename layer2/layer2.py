import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

os.makedirs('outputs', exist_ok=True)

# ── 1. LOAD + PREPROCESS (mirrors ieee_layer1.py exactly) ────────────────────
print("Loading IEEE-CIS dataset...")

train_t = pd.read_csv('datasets/train_transaction.csv')
train_i = pd.read_csv('datasets/train_identity.csv')

df = train_t.merge(train_i, on='TransactionID', how='left')
df = df.drop(columns=['TransactionID'])

thresh = 0.5 * len(df)
df = df.dropna(axis=1, thresh=int(thresh))

X = df.drop(columns=['isFraud'])
y = df['isFraud']

cat_cols = X.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

X = X.fillna(X.median())

scaler = StandardScaler()
for col in ['TransactionAmt', 'TransactionDT']:
    if col in X.columns:
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

# Rebuild full df with processed features
df_processed = X.copy()
df_processed['isFraud'] = y.values

N_FEATURES = X.shape[1]
print(f"Feature count: {N_FEATURES}")

# ── 2. IDENTIFY HARD CASES ────────────────────────────────────────────────────
layer1_model = joblib.load('outputs/ieee_layer1_model.pkl')

fraud_df = df_processed[df_processed['isFraud'] == 1].reset_index(drop=True)
legit_df  = df_processed[df_processed['isFraud'] == 0].reset_index(drop=True)

X_fraud = fraud_df.drop(columns=['isFraud'])
fraud_probs_all = layer1_model.predict_proba(X_fraud)[:, 1]

hard_fraud = fraud_df[fraud_probs_all < 0.80].reset_index(drop=True)
easy_fraud = fraud_df[fraud_probs_all >= 0.80].reset_index(drop=True)

print(f"Hard fraud (held out for eval): {len(hard_fraud)}")
print(f"Easy fraud (used for training): {len(easy_fraud)}")

# ── 3. AUGMENT HARD CASES ────────────────────────────────────────────────────
np.random.seed(42)
hard_fraud_features = hard_fraud.drop(columns=['isFraud'])
augmented_rows = []

for _ in range(20):
    noise = np.random.normal(0, 0.1, hard_fraud_features.shape)
    augmented = hard_fraud_features.copy() + noise
    augmented['isFraud'] = 1
    augmented_rows.append(augmented)

augmented_hard = pd.concat(augmented_rows).reset_index(drop=True)
print(f"Augmented synthetic hard cases: {len(augmented_hard)}")

# ── 4. BUILD TRAINING ENVIRONMENT ────────────────────────────────────────────
df_layer2 = pd.concat([
    easy_fraud,
    augmented_hard,
    legit_df.sample(5000, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nLayer 2 training environment size: {len(df_layer2)}")
print(f"Fraud in training env:             {df_layer2['isFraud'].sum()}")

# ── 5. FRAUD DETECTION ENVIRONMENT ───────────────────────────────────────────
class FraudEnv(gym.Env):
    """
    Custom Gymnasium environment for IEEE-CIS fraud detection.

    Actions:
        0 = Approve
        1 = Flag for review
        2 = Block
    """

    def __init__(self, dataframe, n_features):
        super(FraudEnv, self).__init__()

        self.df = dataframe.reset_index(drop=True)
        self.n_samples = len(self.df)
        self.current_index = 0
        self.n_features = n_features

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(n_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        return self._get_observation(), {}

    def step(self, action):
        true_label = self.df.loc[self.current_index, 'isFraud']
        reward = self._get_reward(action, true_label)

        self.current_index += 1
        terminated = self.current_index >= self.n_samples
        truncated = False

        obs = self._get_observation(last=terminated)
        return obs, reward, terminated, truncated, {}

    def _get_observation(self, last=False):
        idx = self.current_index if not last else self.current_index - 1
        row = self.df.loc[idx].drop('isFraud').values.astype(np.float32)
        return row

    def _get_reward(self, action, true_label):
        if action == 2 and true_label == 1:
            return 3.0      # Correct block — caught fraud
        if action == 0 and true_label == 0:
            return 1.0      # Correct approval
        if action == 1 and true_label == 1:
            return 1.5      # Flagged fraud — acceptable
        if action == 1 and true_label == 0:
            return -0.5     # False alarm
        if action == 2 and true_label == 0:
            return -1.0     # Blocked legitimate
        if action == 0 and true_label == 1:
            return -5.0     # Missed fraud — worst outcome
        return 0.0


# ── 6. TRAIN PPO AGENT ────────────────────────────────────────────────────────
env = FraudEnv(df_layer2, N_FEATURES)

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
model_rl.save("outputs/ieee_layer2_ppo_model")
print("RL model saved to outputs/ieee_layer2_ppo_model")

# ── 7. EVALUATE ON ORIGINAL HARD CASES ───────────────────────────────────────
print("\nEvaluating on original hard fraud cases (never seen during training)...")

hard_legit   = legit_df.sample(500, random_state=99)
hard_eval_df = pd.concat([hard_fraud, hard_legit]).sample(
    frac=1, random_state=99
).reset_index(drop=True)

eval_env = FraudEnv(hard_eval_df, N_FEATURES)
obs, _   = eval_env.reset()
hard_actions, hard_labels = [], []

for i in range(len(hard_eval_df)):
    action, _ = model_rl.predict(obs, deterministic=True)
    hard_actions.append(int(action))
    hard_labels.append(int(hard_eval_df.loc[i, 'isFraud']))
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
if hard_total > 0:
    print(f"Catch rate:                {hard_caught/hard_total:.2%}")