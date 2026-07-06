import json, glob, os
import numpy as np
import pandas as pd

BASE = "//mimer/NOBACKUP/groups/naiss2025-22-1693/Revolve_grid_search/reacher/database/revolve_auto/10/island_0/rl_logs/8_2"
GRID = os.path.join(BASE, "grid_summary_manual.json")


# --- load global fitness table ---
with open(GRID, "r") as f:
    grid = json.load(f)

rows = []

for cfg, g in grid.items():
    cfg_dir = os.path.join(BASE, cfg)
    algo_path = os.path.join(cfg_dir, "algo_cfg.json")
    fin_path  = os.path.join(cfg_dir, "training_final", "final_training.json")

    if not os.path.exists(algo_path):
        continue

    with open(algo_path, "r") as f:
        algo = json.load(f)

    row = {
        "cfg": cfg,
        "fitness": g["total_avg_fitness"],
        "env_max_reward": g["env_max_reward"]["Reacher-v5"],
    }

    # per-seed stats (optional but useful)
    if os.path.exists(fin_path):
        with open(fin_path, "r") as f:
            fin = json.load(f)
        seeds = fin.get("env_seed_stats", {}).get("Reacher-v5", {})
        means = [v["mean_reward"] for v in seeds.values()]
        row["seed_mean_reward_avg"] = np.mean(means) if means else np.nan
        row["seed_mean_reward_std"] = np.std(means) if len(means) > 1 else np.nan

    # flatten algo_cfg
    for k, v in algo.items():
        if isinstance(v, (int, float, bool)):
            row[k] = v

    rows.append(row)

df = pd.DataFrame(rows).sort_values("fitness", ascending=False).reset_index(drop=True)

print("\nTOP 10 CONFIGS")
print(df[["cfg","fitness","env_max_reward","seed_mean_reward_avg"]].head(10).to_string(index=False))

# --- contrast analysis ---
q = 0.2
top = df.head(max(1, int(len(df)*q)))
bot = df.tail(max(1, int(len(df)*q)))

num_cols = [c for c in df.columns if c not in
            ["cfg","fitness","env_max_reward","seed_mean_reward_avg","seed_mean_reward_std"]]

rows = []
for c in num_cols:
    if df[c].dtype.kind not in "if": 
        continue
    rows.append({
        "param": c,
        "top_median": top[c].median(),
        "bot_median": bot[c].median(),
        "median_diff": top[c].median() - bot[c].median(),
        "spearman_r": df[c].corr(df["fitness"], method="spearman")
    })

stats = pd.DataFrame(rows).sort_values("median_diff", ascending=False)

print("\nTOP–BOTTOM PARAM SHIFTS")
print(stats.head(15).to_string(index=False))

print("\nHIGHEST CORRELATIONS")
print(stats.sort_values("spearman_r", ascending=False).head(15).to_string(index=False))

df.to_csv("cfg_table.csv", index=False)
print("\nSaved cfg_table.csv")
