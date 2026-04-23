## FinGuard

### Layer 1 — Done
- Run: python3 layer1/main.py
- Output: outputs/layer1_model.pkl, layer1_misses.csv, layer2_candidates.csv

### Layer 2 — In Progress
- Input: outputs/layer1_misses.csv + outputs/layer1_model.pkl
- Task: RL environment that catches what Layer 1 missed
- See layer2/ folder

### Dataset
- Download creditcard.csv from Kaggle and place in datasets/
- Not pushed to repo (too large)