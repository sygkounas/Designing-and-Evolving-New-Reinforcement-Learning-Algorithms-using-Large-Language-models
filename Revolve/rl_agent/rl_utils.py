#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


############################################################
# Load metrics with EventAccumulator (correct version)
############################################################

############################################################
# Save final fitness score as text for easy loading
############################################################
def save_fitness_score(algo_dir: Path, tag: str, value: float):
    """
    Saves the average fitness score (0–1) into algo_dir/fitness_score_<tag>.txt
    Example: fitness_score_first.txt or fitness_score_final.txt
    """
    out_path = Path(algo_dir) / f"fitness_score_{tag}.txt"
    out_path.write_text(f"{value}\n")



def load_all_metrics(run_dir):
    """
    Returns a dict containing all events for:
    - eval/mean_reward
    - train/loss
    - train/grad_norm
    - train/param_norm

    Structure:
    {
        "eval/mean_reward": [(step, value), ...],
        "train/loss": [...],
        "train/grad_norm": [...],
        "train/param_norm": [...]
    }
    """

    acc = event_accumulator.EventAccumulator(str(run_dir))
    acc.Reload()

    keys = {
        "eval/mean_reward": None,
        "train/loss": None,
        "train/grad_norm": None,
        "train/param_norm": None,
    }

    out = {k: [] for k in keys.keys()}

    for key in keys.keys():
        if key in acc.scalars.Keys():
            events = acc.scalars.Items(key)
            out[key] = [(e.step, float(e.value)) for e in events]

    return out


############################################################
# Helper: evenly sample N points (value-only list)
############################################################
def sample_n_points(values, n_points):
    """
    values: list of floats
    returns: list of exactly n_points (or fewer if len< n_points)
    """
    if len(values) == 0:
        return []

    if len(values) <= n_points:
        return [float(v) for v in values]

    idxs = np.linspace(0, len(values) - 1, n_points, dtype=int)
    return [float(values[i]) for i in idxs]


############################################################
# Collect metrics from 0 → best reward step
############################################################
def collect_tb_metrics_to_json(algo_dir: Path, phase: str, algo_idx: int, n_points: int):
    """
    Improved logic:
      - If no reward events => fallback to uniform sampling of all traces
      - If best_step == 0 => also fallback to uniform sampling (do NOT slice everything away)
      - Otherwise slice traces up to best_step and sample n_points values.
    """

    phase_dir = algo_dir / phase
    results = {}

    for run_dir in sorted(phase_dir.iterdir()):

        if not run_dir.is_dir():
            continue

        env_name = run_dir.name
        results[env_name] = {}

        for seed_dir in sorted(run_dir.glob("seed_*")):

            # Load all raw events for the 4 metrics
            metrics_raw = load_all_metrics(seed_dir)
            reward_events = metrics_raw["eval/mean_reward"]

            # ----------------------------------------
            # CASE 1: No reward events at all
            # ----------------------------------------
            if len(reward_events) == 0:
                loss_vals   = [v for (_, v) in metrics_raw["train/loss"]]
                grad_vals   = [v for (_, v) in metrics_raw["train/grad_norm"]]
                param_vals  = [v for (_, v) in metrics_raw["train/param_norm"]]

                sampled_loss   = sample_n_points(loss_vals,  n_points)
                sampled_grad   = sample_n_points(grad_vals,  n_points)
                sampled_param  = sample_n_points(param_vals, n_points)

                # MountainCar might truly have no reward signal
                sampled_reward = [0.0] * n_points

                results[env_name][seed_dir.name] = {
                    "eval_reward": sampled_reward,
                    "train_loss":  sampled_loss,
                    "grad_norm":   sampled_grad,
                    "param_norm":  sampled_param
                }
                continue

            # ----------------------------------------
            # CASE 2: Reward events exist
            # Compute best_step
            # ----------------------------------------
            reward_values = [v for (_, v) in reward_events]
            idx_best = int(np.argmax(reward_values))
            best_step = idx_best * 5000  # eval idx → step mapping

            # ----------------------------------------
            # CASE 2a: best_step == 0 → fallback to full uniform sampling
            # (prevents slicing everything away)
            # ----------------------------------------
            if best_step == 0:
                loss_vals   = [v for (_, v) in metrics_raw["train/loss"]]
                grad_vals   = [v for (_, v) in metrics_raw["train/grad_norm"]]
                param_vals  = [v for (_, v) in metrics_raw["train/param_norm"]]
                reward_vals = [v for (_, v) in reward_events]

                results[env_name][seed_dir.name] = {
                    "eval_reward": sample_n_points(reward_vals,  n_points),
                    "train_loss":  sample_n_points(loss_vals,    n_points),
                    "grad_norm":   sample_n_points(grad_vals,    n_points),
                    "param_norm":  sample_n_points(param_vals,   n_points),
                }
                continue

            # ----------------------------------------
            # CASE 2b: Normal slicing up to best_step
            # ----------------------------------------
            def slice_metric(ev_list):
                return [v for (step, v) in ev_list if step <= best_step]

            sliced_reward = slice_metric(reward_events)
            sliced_loss   = slice_metric(metrics_raw["train/loss"])
            sliced_grad   = slice_metric(metrics_raw["train/grad_norm"])
            sliced_param  = slice_metric(metrics_raw["train/param_norm"])

            sampled_reward = sample_n_points(sliced_reward, n_points)
            sampled_loss   = sample_n_points(sliced_loss,   n_points)
            sampled_grad   = sample_n_points(sliced_grad,   n_points)
            sampled_param  = sample_n_points(sliced_param,  n_points)

            results[env_name][seed_dir.name] = {
                "eval_reward": sampled_reward,
                "train_loss":  sampled_loss,
                "grad_norm":   sampled_grad,
                "param_norm":  sampled_param,
            }

            print(f"[DEBUG] {env_name}/{seed_dir.name}: best_step={best_step}, "
                  f"len_reward={len(sampled_reward)}")

    return results


############################################################
# Fitness summarization (unchanged)
############################################################
def summarize_fitness(algo_dir: Path, n_points: int, phase: str):
    """
    Uses MAX reward per seed (same logic as before).
    """

    phase_dir = algo_dir / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"Missing training directory: {phase_dir}")

    env_raw = {}
    env_fit = {}

    # Same reward bounds as before
    BOUNDS = {
        "CartPole-v1": (0, 500),
        "MountainCar-v0": (-200, -70),
        "Acrobot-v1": (-500, 0),
       # "Pendulum-v1": (-2000, 0),
        "LunarLander-v3": (0, 300),
        "HalfCheetah-v5":   (0, 7000),     # added

    }

    for run_dir in sorted(phase_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        env_name = run_dir.name

        seed_maxes = []

        for seed_dir in sorted(run_dir.glob("seed_*")):

            metrics_raw = load_all_metrics(seed_dir)
            reward_events = metrics_raw["eval/mean_reward"]

            if len(reward_events) == 0:
                seed_maxes.append(-100000.0)
                continue

            reward_values = [v for (_, v) in reward_events]
            seed_maxes.append(max(reward_values))

        avg_raw = float(np.mean(seed_maxes)) if seed_maxes else 0.0
        env_raw[env_name] = avg_raw

        lo, hi = BOUNDS.get(env_name, (0, 1))
        if hi != lo:
            norm = (avg_raw - lo) / (hi - lo)
            env_fit[env_name] = float(np.clip(norm, 0.0, 1.0))
        else:
            env_fit[env_name] = 0.0

    total_fit = float(np.mean(list(env_fit.values()))) if env_fit else 0.0

    return {
        "env_max_reward": env_raw,
        "env_fitness": env_fit,
        "total_avg_fitness": total_fit,
    }


