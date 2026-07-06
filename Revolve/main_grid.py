import os
import sys
import traceback
sys.path.append(os.environ["ROOT_PATH"])
#from rewards_database import RevolveDatabase
#from modules import *
#import utils
#import prompts
#from evolutionary_utils.custom_environment import CustomEnvironment
import absl.logging as logging
from functools import partial
#from utils import *
import hydra
import os
import glob
from typing import Callable, List
#from rl_agent.main_rl import run_rl_algorithm
from rl_agent.run_main import run_rl_algorithm


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.environ["ROOT_PATH"], "cfg"),
    config_name="generate",
)
def main(cfg):
    # create log directory
    log_dir = cfg.data_paths.output_logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
   # tracker = utils.DataLogger(os.path.join(log_dir, "progress.log"))


    #for generation_id in range(cfg.evolution.star_generation, cfg.evolution.num_generations):
    for generation_id in range(8, 9):
        operators = []
        parent_fn_paths_list = []

    
        # fix the temperature for sampling
        print(
            f"\n========= Generation {generation_id} | Model: {cfg.evolution.baseline} =========="
        )
        # load all groups if iteration_id > 0, else initialize empty islands

        rew_fn_strings = []  # valid rew fns
        # fitness_scores = []
        island_ids = []
        counter_ids = []
        # metrics_dicts = []
        policies = []
        parents_list = []
        load=1


        # for each generation, produce new individuals via mutation or crossover
        for counter_id in range(2,3):
            if load==0:
                pass
            else:
                # -------------------------------------------------
                # LOAD EXISTING INDIVIDUALS FROM CORRECT ISLAND
                # -------------------------------------------------
                # Path: database/<baseline>/<run_id>/island_*/generated_fns/GEN_COUNTER.txt
                # Example:  database/revolve/1/island_0/generated_fns/1_2.txt
                # -------------------------------------------------
                base_root = os.path.join(
                    os.environ["ROOT_PATH"],
                    "database",
                    cfg.evolution.baseline,
                    str(cfg.data_paths.run)
                )

                # Search all islands for the reward file GEN_COUNTER.txt
                pattern = os.path.join(base_root, "island_*", "generated_fns",
                                    f"{generation_id}_{counter_id}.txt")

                matches = glob.glob(pattern)

                if len(matches) == 0:
                    logging.info(f"NO existing reward function found for {generation_id}_{counter_id}")
                    continue
                if len(matches) > 1:
                    logging.warning(f"Multiple matches for {generation_id}_{counter_id}, taking first.")

                reward_path = matches[0]

                # Extract island id from folder name
                island_id = int(os.path.basename(os.path.dirname(os.path.dirname(reward_path)))[7:])

                # Load reward function code
                with open(reward_path, "r") as f:
                    reward_func_str = f.read()

                logging.info(f"Loaded existing reward fn from: {reward_path} (island {island_id})")
                reward_base = os.path.join(
                    os.environ["ROOT_PATH"],
                    "database",
                    cfg.evolution.baseline,
                    str(cfg.data_paths.run),
                    f"island_{island_id}"
                )
            rew_fn_strings.append(reward_func_str)
            counter_ids.append(counter_id)
           # print("reward_func_str",reward_func_str)
            
            try:
              #  fitness, metrics = run_rl_algorithm(reward_func_str,generation_id, counter_id, island_id, reward_base)
                fitness, info = run_rl_algorithm(
                rl_code=reward_func_str,
                generation_id=str(generation_id),
                counter_id=str(counter_id),
                island_id=str(island_id),

                base_out_dir=reward_base,

                # --- CFG PARALLELIZATION ---
                n_cfgs_total=cfg.grid_search.n_cfgs_total,
                cfg_seed=cfg.grid_search.cfg_seed,
                cfg_start=cfg.grid_search.cfg_start,
                cfg_end=cfg.grid_search.cfg_end,

                device=cfg.grid_search.device,
                n_procs=cfg.grid_search.n_procs,
            )
            except Exception as e:
                logging.info(f"RL algo failed: {e}")
                continue   

        


if __name__ == "__main__":
    main()
