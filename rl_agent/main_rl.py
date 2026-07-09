# main_rl.py
# Pure Python module — exposes run_rl_algorithm()

import gc
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import json
import multiprocessing as mp

from rl_agent.rl_utils import summarize_fitness, collect_tb_metrics_to_json

# -----------------------
# CONFIG
# -----------------------
ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1",
        "HalfCheetah-v5", "LunarLander-v3"]
SEEDS = [0, 1]
TOTAL_STEPS = 1_000_000
N_JSON = 100


# ---------------------------------------------------------
# TOP-LEVEL TRAIN FUNCTION (SAFE FOR multiprocessing.spawn)
# ---------------------------------------------------------
def _train_once(env_name, algo_class, out_dir: Path, seed: int, device):
    """Train one env × seed."""
    run_dir = out_dir / "training_final" / env_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    try:
        env.reset(seed=seed)
    except Exception:
        pass

    model = algo_class(env, verbose=0, log_dir=str(run_dir), device=device)
    model.learn(total_timesteps=TOTAL_STEPS)

    env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------
# TOP-LEVEL JOB WRAPPER (MUST BE HERE — NOT NESTED!)
# ---------------------------------------------------------
def _job(env_name, seed, out_dir, device_str, algo_code):
    """
    Each process executes this. Recreates NewAlgo inside the subprocess.
    """
    device = torch.device(device_str)

    gbl = {}
    exec(algo_code, gbl)
    algo_class = gbl["NewAlgo"]

    _train_once(env_name, algo_class, Path(out_dir), seed, device)


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
def run_rl_algorithm(rl_code, generation_id, counter_id, island_id, base_out_dir: str):

    # 1. Exec RL code in main process to check validity
    gbl = {}
    exec(rl_code, gbl)
    if "NewAlgo" not in gbl:
        raise RuntimeError("RL code must define class NewAlgo(env, ...).")

    # 2. Output directory
    out_dir = (
        Path(base_out_dir)
        / "rl_logs"
        / f"{generation_id}_{counter_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Prepare parallel tasks (5 envs × 2 seeds = 10 tasks)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tasks = [
        (env_name, seed, str(out_dir), str(device), rl_code)
        for env_name in ENVS
        for seed in SEEDS
    ]

    # 4. Number of processes — EXACT SAME LOGIC AS gpu.py
    n_procs = min(len(tasks), mp.cpu_count())

    # 5. Parallel execution
    with mp.get_context("spawn").Pool(processes=n_procs) as pool:
        pool.starmap(_job, tasks)

    # 6. Summaries & fitness
    fitness_dict = summarize_fitness(out_dir, N_JSON, phase="training_final")
    fitness = float(fitness_dict.get("total_avg_fitness", 0.0))

    (out_dir / "training_final" / "final_training.json").write_text(
        json.dumps(fitness_dict, indent=2)
    )

    metrics = collect_tb_metrics_to_json(
        out_dir, "training_final", algo_idx=0, n_points=N_JSON
    )

    return fitness, metrics
