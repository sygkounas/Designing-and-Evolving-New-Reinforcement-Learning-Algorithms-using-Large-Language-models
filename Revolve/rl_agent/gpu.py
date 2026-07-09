#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, sys, time, json, traceback, multiprocessing as mp
from pathlib import Path
import numpy as np
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
import gymnasium as gym
import torch, gc, warnings, os

# --- local utils ---
from utils import summarize_fitness, collect_tb_metrics_to_json, save_fitness_score

# ---------------- CONFIG ----------------
MODEL = "gpt-5"
BASE_DIR = Path(__file__).parent
PROMPT_DIR = BASE_DIR / "prompts"
INIT_PATH = PROMPT_DIR / "init.txt"
ENV_PATH = PROMPT_DIR / "env.txt"
REFINE_PATH = PROMPT_DIR / "refine_env.txt"
REFINE_WITH_RESULTS_PATH = PROMPT_DIR / "refine_env_final.txt"

RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

LOAD_ONLY = 0
EXIST_ALGO_START = 9
N_ALGORITHMS = 3

#ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1", "LunarLander-v2"]
ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "HalfCheetah-v5", "LunarLander-v3"]

SEEDS = [0, 1]
N_POINTS_JSON = 100
PARALLEL_PROCS = min(mp.cpu_count(), len(ENVS) * len(SEEDS))


# ---------------- HELPERS ----------------
def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def call_model(client, prompt: str, stage_name: str, save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    input_file = save_dir / f"{stage_name}_input.txt"
    output_file = save_dir / f"{stage_name}_output.txt"
    input_file.write_text(prompt, encoding="utf-8")
    resp = client.chat.completions.create(model=MODEL,
                                          messages=[{"role": "user", "content": prompt}])
    out = resp.choices[0].message.content
    output_file.write_text(out, encoding="utf-8")
    return out


def extract_algo_block(text: str, stage: str, code_dir: Path) -> str:
    match = re.search(r"(?:```python|python\s*''')\s*(.*?)\s*(?:```|''')",
                      text, re.DOTALL)
    stage_map = {
        "init_gen": "init",
        "refine_1": "refine_1",
        "refine_results": "refine_final",
    }
    fname = stage_map.get(stage, stage)
    code_dir.mkdir(exist_ok=True, parents=True)
    out_path = code_dir / f"{fname}.py"
    if not match:
        out_path.write_text("# extraction failed\n")
        raise ValueError(f"No python code block in stage={stage}")
    code = match.group(1).strip()
    out_path.write_text(code, encoding="utf-8")
    return code


# ---------------- TRAIN EXEC ----------------
def _run_single(env_name, algo_code, algo_dir_str, rel, seed, steps):
    device = torch.device("cuda:0")
    algo_dir = Path(algo_dir_str)
    try:
        gbl = {}
        exec(algo_code, gbl)
        NewAlgo = gbl["NewAlgo"]

        np.random.seed(seed)
        torch.manual_seed(seed)
        env = gym.make(env_name)
        try:
            env.reset(seed=seed)
        except:
            pass

        run_dir = algo_dir / rel
        run_dir.mkdir(exist_ok=True, parents=True)

        model = NewAlgo(env, verbose=0, log_dir=str(run_dir), device=device)
        model_path = run_dir / "checkpoints" / "final_model.pt"

        print(f"[TRAIN] {rel} | env={env_name} seed={seed}")
        model.learn(total_timesteps=steps)
        model.save(str(model_path))

        env.close() 
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        return {"ok": True}

    except Exception as e:
        err = traceback.format_exc()
        (algo_dir / f"error_{env_name}_seed{seed}.log").write_text(err)
        print(f"[ERROR] during {env_name} seed {seed}: {err.splitlines()[-1]}")
        return {"ok": False}


def run_many(envs, seeds, algo_code, algo_dir, tag, steps):
    tasks = []
    for e in envs:
        for s in seeds:
            rel = f"{tag}/{e}/seed_{s}"
            tasks.append((e, algo_code, str(algo_dir), rel, s, steps))

    print(f"[PARALLEL] Running {tag}")
    with mp.get_context("spawn").Pool(processes=PARALLEL_PROCS) as pool:
        pool.starmap(_run_single, tasks)
    print(f"[PARALLEL] Done {tag}")


# ---------------- MAIN ----------------
def main():
    #client = OpenAI()
    client = OpenAI(api_key="")

    init_prompt = load_prompt(INIT_PATH)
    env_prompt = load_prompt(ENV_PATH)
    refine_prompt = load_prompt(REFINE_PATH)
    refine_results_prompt = load_prompt(REFINE_WITH_RESULTS_PATH)

    for algo_idx in range(EXIST_ALGO_START, EXIST_ALGO_START + N_ALGORITHMS):

        algo_dir = RUNS_DIR / f"algo_{algo_idx}"
        code_dir = algo_dir / "code"
        fitness_dir = algo_dir / "fitness"
        results_dir = algo_dir / "results"

        algo_dir.mkdir(exist_ok=True, parents=True)
        code_dir.mkdir(exist_ok=True, parents=True)
        fitness_dir.mkdir(exist_ok=True, parents=True)
        results_dir.mkdir(exist_ok=True, parents=True)

        errors = {}
        final_algo_path = algo_dir / "final_algorithm.py"

        # ------------------------------------------------------------------
        # LOAD_ONLY MODE — load v02 for first training, v03 for final training
        # ------------------------------------------------------------------
        if LOAD_ONLY:
            v02_path = code_dir / "refine_1.py"
            v03_path = code_dir / "refine_final.py"

            if not v02_path.exists():
                raise FileNotFoundError(f"Missing v02: {v02_path}")
            if not v03_path.exists():
                raise FileNotFoundError(f"Missing v03: {v03_path}")

            algo_code_v02 = v02_path.read_text()
            algo_code_v03 = v03_path.read_text()

        else:
            # ---------------- LLM PIPELINE (unchanged) ----------------
            combined = f"{init_prompt}\n\n{env_prompt}"
            out = call_model(client, combined, "init_gen", code_dir)
            algo_code = extract_algo_block(out, "init_gen", code_dir)

            inp = refine_prompt.replace("{Your_Algorithm}", algo_code)
            out = call_model(client, inp, "refine_1", code_dir)
            algo_code_v02 = extract_algo_block(out, "refine_1", code_dir)

        # ===================== TRAINING FIRST (v02) =====================
        run_many(ENVS, SEEDS, algo_code_v02, algo_dir, "training_first", 1_000_000)

        first_fit = summarize_fitness(algo_dir, N_POINTS_JSON, "training_first")
        (fitness_dir / "first_training.json").write_text(json.dumps(first_fit, indent=2))
        if "total_avg_fitness" in first_fit:
            save_fitness_score(algo_dir, "first", first_fit["total_avg_fitness"])

       # results1 = collect_tb_metrics_to_json(algo_dir, "training_first", algo_idx, N_POINTS_JSON)
        results1 = collect_tb_metrics_to_json(
            algo_dir,
            "training_first",
            algo_idx,
            N_POINTS_JSON,
        )
        (results_dir / "metrics_first_training.json").write_text(json.dumps(results1, indent=2))

        # ===================== TRAINING FINAL (v03) =====================
        if not LOAD_ONLY:
            inp = refine_results_prompt.replace("{Your_Algorithm}", algo_code_v02)
            inp = inp.replace("{Results}", json.dumps(results1)).replace("{Errors}", json.dumps(errors))
            out = call_model(client, inp, "refine_results", code_dir)
            algo_code_v03 = extract_algo_block(out, "refine_results", code_dir)

        run_many(ENVS, SEEDS, algo_code_v03, algo_dir, "training_final", 1_000_000)

        final_fit = summarize_fitness(algo_dir, N_POINTS_JSON, "training_final")
        (fitness_dir / "final_training.json").write_text(json.dumps(final_fit, indent=2))
        if "total_avg_fitness" in final_fit:
            save_fitness_score(algo_dir, "final", final_fit["total_avg_fitness"])

        results2 = collect_tb_metrics_to_json(
    algo_dir,
    "training_final",
    algo_idx,
    N_POINTS_JSON,
)
        (results_dir / "metrics_final_training.json").write_text(json.dumps(results2, indent=2))

        final_algo_path.write_text(algo_code_v03, encoding="utf-8")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


# Fix (short, exact, safe)
# STEP 1 — IN env.txt (the base template the LLM copies)

# Find:

# DEVICE = "cpu"


# Change to:

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This is safe for:

# GPU machines

# CPU-only machines (fallback)

# Distributed or multiprocess

# Any model the LLM generates

# Nothing else required in this file.

# Why this fix works

# Your LLM-generated algorithm always uses:

# .to(DEVICE)
# torch.as_tensor(..., device=DEVICE)


# So changing the definition of DEVICE at the top of the base template automatically pushes everything to GPU, without requiring any further changes.

# It also avoids:

# Breaking mp.Pool

# Hardcoding “cuda” when GPU might not exist

# Manual edits inside the generated model

# STEP 2 — OPTIONAL (safer long-term)

# In init.txt (

# init

# ), add ONE sentence to the rules section:

# Add just this line:

# The generated code must always use DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") for selecting the compute device.

# This ensures future LLM generations won’t override it back to CPU.

# Final answer (super short):
# Modify in env.txt:

# FROM:

# DEVICE = "cpu"


# TO:

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# nvidia-smi dmon -s u