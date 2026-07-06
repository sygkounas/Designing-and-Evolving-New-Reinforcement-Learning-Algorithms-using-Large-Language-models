from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(
    "/home/alkis/Downloads/new_rl_stuff/labbet_runs/Revolve/Revolve/"
    "database/revolve_auto/10/island_0/rl_logs/9_4"
)

ENV = "Acrobot-v1"

BOUNDS = {
    "CartPole-v1": (0, 500),
    "MountainCar-v0": (-200, -70),
    "Acrobot-v1": (-500, 0),
    "LunarLander-v3": (0, 300),
    "HalfCheetah-v5": (0, 7000),
}


def load_rewards(seed_dir: Path):
    acc = event_accumulator.EventAccumulator(str(seed_dir))
    acc.Reload()
    if "eval/mean_reward" not in acc.scalars.Keys():
        return []
    return [v.value for v in acc.scalars.Items("eval/mean_reward")]


def normalize(val, env):
    lo, hi = BOUNDS[env]
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


rows = []

for cfg_dir in sorted(ROOT.glob("cfg_*")):
    env_dir = cfg_dir / "training_final" / ENV
    if not env_dir.exists():
        continue

    seed_stats = {}
    seed_maxes = []

    for seed_dir in sorted(env_dir.glob("seed_*")):
        rewards = load_rewards(seed_dir)

        if len(rewards) == 0:
            mean_r = 0.0
            max_r = -1e9
        else:
            mean_r = float(np.mean(rewards))
            max_r = float(np.max(rewards))

        fitness = normalize(max_r, ENV)

        seed_stats[seed_dir.name] = {
            "mean_reward": mean_r,
            "max_reward": max_r,
            "fitness": fitness,
        }

        seed_maxes.append(max_r)

    if not seed_maxes:
        continue

    avg_max = float(np.mean(seed_maxes))
    avg_fitness = normalize(avg_max, ENV)

    rows.append({
        "cfg": cfg_dir.name,
        "avg_max_reward": avg_max,
        "avg_fitness": avg_fitness,
        "seed_stats": seed_stats,
    })


# sort best → worst
rows.sort(key=lambda x: x["avg_fitness"], reverse=True)

TOP_K = 5
top = rows[:TOP_K]


# ---------------- PRINT TOP 5 ----------------
for rank, r in enumerate(top, 1):
    print("=" * 72)
    print(f"RANK #{rank}")
    print("CFG:", r["cfg"])
    print("AVG MAX REWARD:", round(r["avg_max_reward"], 3))
    print("AVG FITNESS:", round(r["avg_fitness"], 4))

    for seed, stats in r["seed_stats"].items():
        print(
            f"  {seed}: "
            f"mean={stats['mean_reward']:.2f}, "
            f"max={stats['max_reward']:.2f}, "
            f"fitness={stats['fitness']:.4f}"
        )
