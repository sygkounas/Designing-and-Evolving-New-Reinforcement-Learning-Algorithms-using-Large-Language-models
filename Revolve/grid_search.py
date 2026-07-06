import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from pathlib import Path
from rl_agent.main_rl import run_rl_algorithm

ALGO_PATH = "/home/alkis/Downloads/new_rl_stuff/labbet_runs/Revolve/Revolve/database/revolve_auto/10/island_0/generated_fns/9_4.txt"
BASE_OUT_DIR = "/home/alkis/Downloads/new_rl_stuff/labbet_runs/Revolve/Revolve"

GENERATION_ID = "10"
DEVICE = "cuda:0"
N_PROCS_INNER = 10      # env × seed parallelism (the one you KNOW works)
N_CFGS_PARALLEL = 2     # THIS is what you asked for


def run_one_cfg(cfg_idx: int):
    rl_code = Path(ALGO_PATH).read_text()

    counter_id = f"cartpole_cfg_{cfg_idx}"

    fitness, _ = run_rl_algorithm(
        rl_code=rl_code,
        generation_id=GENERATION_ID,
        counter_id=counter_id,
        island_id="0",
        base_out_dir=BASE_OUT_DIR,
    )

    print(f"[CFG {cfg_idx}] fitness = {fitness}")
    return fitness


if __name__ == "__main__":
    cfg_ids = list(range(N_CFGS_PARALLEL))

    with mp.Pool(processes=N_CFGS_PARALLEL) as pool:
        pool.map(run_one_cfg, cfg_ids)
