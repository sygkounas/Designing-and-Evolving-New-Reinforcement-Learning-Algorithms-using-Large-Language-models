import os
import sys
import traceback
sys.path.append(os.environ["ROOT_PATH"])
from rewards_database import RevolveDatabase
from modules import *
import utils
import prompts
from evolutionary_utils.custom_environment import CustomEnvironment
import absl.logging as logging
from functools import partial
from utils import *
import hydra
import os
import glob
from typing import Callable, List
from rl_agent.main_rl import run_rl_algorithm

def load_reward_function(file_path: str) -> Callable:
    """
    Load and return a callable reward function from a file.
    Args:
    - file_path (str): Path to the file containing the reward function.

    Returns:
    - Callable: Executable reward function.
    """
    with open(file_path, "r") as f:
        reward_fn_str = f.read()

    # Use define_function_from_string to make it executable
    reward_func, _ = define_function_from_string(reward_fn_str)

    if reward_func is None:
        raise ValueError("Failed to load a valid reward function.")

    return reward_func


def is_valid_reward_fn(generated_fn: Callable, generated_fn_str: str, args: List[str]):
    """validate generated heuristic function"""
    if generated_fn is None or args is None:
        raise utils.InvalidFunctionError("Generated function has no arguments.")
    env_state = CustomEnvironment().env_state
    env_vars = env_state.keys()
    # check if all args are valid env args
    if set(args).intersection(set(env_vars)) != set(args):
        raise utils.InvalidFunctionError("Generated function uses invalid arguments.")
    # TODO: test the following for REvolve
    # Get the return type annotation
    return_statements = utils.validate_callable_no_signature(generated_fn_str)
    if not return_statements:
        raise utils.InvalidFunctionError(
            "The function does not have any return statements."
        )
    return True


def generate_valid_reward(
        reward_generation: RewardFunctionGeneration,
        in_context_prompt: str,
        max_trials: int = 10,
) -> [str, List[str]]:
    """
    single function generation until valid
    :param reward_generation: initialized class of RewardFunctionGeneration
    :param in_context_prompt: in context prompt used by the LLM to generate the new fn
    :param max_trials: maximum number of trials to generate
    :return: return valid function string
    """
    # used in case we want to provide python error feedbacks to the LLM
    error_feedback = ""
    error_flag = False
    trials = 0
    while True:
        try:
            rew_func_str = reward_generation.generate_rf(
                in_context_prompt + error_feedback
            )
            rew_func, args = utils.define_function_from_string(rew_func_str)
            is_valid_reward_fn(rew_func, rew_func_str, args)
            logging.info("Valid reward function generated.")
            error_flag = False
            error_feedback = ""
            break  # Exit the loop if successful
        except Exception as e:
            logging.info(f"Specific error caught: {e}")
        logging.info("Attempting to generate a new function due to an error.")
        trials += 1
        if trials >= max_trials:
            logging.info("Exceeded max trials.")
            return None, None, None
    return rew_func_str, args


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.environ["ROOT_PATH"], "cfg"),
    config_name="generate",
)
def main(cfg):
    env_name = cfg.environment.name

    system_prompt = prompts.types["system_prompt"]
    env_prompt = prompts.types["env_prompt"]
   # env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(
        system_prompt=system_prompt
    )

    # create log directory
    log_dir = cfg.data_paths.output_logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tracker = utils.DataLogger(os.path.join(log_dir, "progress.log"))

    # define a schedule for temperature of sampling
    temp_scheduler = partial(
        utils.linear_decay,
        initial_temp=cfg.database.initial_temp,
        final_temp=cfg.database.final_temp,
        num_iterations=cfg.evolution.num_generations,
    )
    if "revolve_auto" in cfg.evolution.baseline:
        database = partial(
            RevolveDatabase,
            num_islands=cfg.database.num_islands,
            max_size=cfg.database.max_island_size,
            crossover_prob=cfg.database.crossover_prob,
            migration_prob=cfg.database.migration_prob,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )
    operator = "seed"
    parent_paths = []

    #for generation_id in range(cfg.evolution.star_generation, cfg.evolution.num_generations):
    for generation_id in range(9, 10):
        operators = []
        parent_fn_paths_list = []

    
        # fix the temperature for sampling
        temperature = temp_scheduler(iteration=generation_id)
        print(
            f"\n========= Generation {generation_id} | Model: {cfg.evolution.baseline} | temperature: {round(temperature, 2)} =========="
        )
        # load all groups if iteration_id > 0, else initialize empty islands
        rewards_database = database(load_islands=not generation_id == 0)

        rew_fn_strings = []  # valid rew fns
        # fitness_scores = []
        island_ids = []
        counter_ids = []
        # metrics_dicts = []
        policies = []
        parents_list = []
        load=1

        # for each generation, produce new individuals via mutation or crossover
        for counter_id in range(4,5):
            if load==0:
                if generation_id == 0:  # initially, uniformly populate the islands
                    # TODO: to avoid corner cases, populate all islands uniformly
                    island_id = random.choice(range(rewards_database.num_islands))
                    in_context_samples = (None, None)
                    operator_prompt = ""
                    logging.info(
                        f"Generation {generation_id}, Counter {counter_id}: island_id={island_id}, type={type(island_id)}"
                    )
                #    parent_list.append({"parents": []})


                else:  # gen_id > 0: start the evolutionary process
                    (
                        in_context_samples,
                        island_id,
                        operator,
                    ) = rewards_database.sample_in_context(
                        cfg.few_shot, temperature
                    )  # weighted sampling of islands and corresponding individuals
                    # operator is now: 'mutation_micro', 'mutation_macro', or 'crossover'
                    print("reward database returned:", in_context_samples, island_id, operator)
                    op_key = operator
                    if "auto" in cfg.evolution.baseline:
                        op_key = f"{operator}_auto"
                    operator_prompt = prompts.types[op_key]


                    # ----- lineage: parse parent (and possibly second parent) from file names -----
                    parents: List[tuple] = []

                    def _parse_gen_counter_from_path(path: str) -> tuple:
                        """
                        path: .../generated_fns/<gen>_<counter>.txt
                        returns (gen, counter) as ints
                        """
                        base = os.path.basename(path)
                        stem = os.path.splitext(base)[0]  # "GEN_COUNTER"
                        gen_str, cnt_str = stem.split("_")
                        return int(gen_str), int(cnt_str)

                    if operator.startswith("mutation"):
                        # single parent = first in-context example
                        if in_context_samples and in_context_samples[0][0] is not None:
                            parents.append(_parse_gen_counter_from_path(in_context_samples[0][0]))
                    elif operator == "crossover":
                        # two parents = first two in-context examples
                        if len(in_context_samples) >= 1 and in_context_samples[0][0] is not None:
                            parents.append(_parse_gen_counter_from_path(in_context_samples[0][0]))
                        if len(in_context_samples) >= 2 and in_context_samples[1][0] is not None:
                            parents.append(_parse_gen_counter_from_path(in_context_samples[1][0]))

                    parents_list.append(parents)
                    parent_paths = [fp for (fp, _) in in_context_samples]  # 1 or 2


                    island_ids.append(island_id)
                    operators.append(operator)
                    parent_fn_paths_list.append(parent_paths)
                # each sample in 'in_context_samples' is a tuple of (fn_path: str, fitness_score: float)
                reward_base = os.path.join(
                    os.environ["ROOT_PATH"],
                    "database",
                    cfg.evolution.baseline,
                    str(cfg.data_paths.run),
                    f"island_{island_id}"
                )
            if load==0:
                base_prompt = (
                    system_prompt + "\n\n" + env_prompt
                    if generation_id == 0
                    else system_prompt + "\n\n" + operator_prompt
                )

                in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(
                    in_context_samples,
                    base_prompt,
                    evolve=generation_id > 0,
                    baseline=cfg.evolution.baseline,
                )
                # in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(
                #     in_context_samples,
                #     operator_prompt,
                #     evolve=generation_id > 0,
                #     baseline=cfg.evolution.baseline,
                # )
                logging.info(f"Designing RL algorithm for counter {counter_id}")
              #  print("In-context prompt prepared.",in_context_prompt)
                # 1. Call LLM
                model_output = call_model(
                    client,
                    in_context_prompt,
                    save_dir=Path(reward_base) / f"llm_gen_{generation_id}_{counter_id}"
                )
                #print("LLM output:", model_output)

                # 2. Extract the python code block
                try:
                    reward_func_str = extract_algo_block(model_output)
                   # print("Extracted reward function string:", reward_func_str)
                except Exception as e:
                    logging.info(f"Failed to extract python block: {e}")
                    continue
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
            print("reward_func_str",reward_func_str)
            
            try:
                fitness, metrics = run_rl_algorithm(reward_func_str,generation_id, counter_id, island_id, reward_base)
            except Exception as e:
                logging.info(f"RL algo failed: {e}")
                continue   

        
            logging.info("Evaluation finished.")
            fitness_scores   = [fitness]         # list of length 1
            metrics_dicts    = [metrics]         # list of length 1
            gen_ids          = [generation_id]   # list of length 1
            ctr_ids          = [counter_id]      # list of length 1
            rew_list         = [reward_func_str] # list of length 1
            isl_ids          = [island_id]       # list of length 1
            parents_for_one = parent_paths      # parent_paths is already a LIST
            operator_list   = [operator]        # wrap operator in a list
            parent_paths_list = [parents_for_one]  

            # store individuals only if it improves overall island fitness
            # for initialization, we don't use this step
            if generation_id == 0:
                rewards_database.seed_islands(
                    gen_ids,
                    ctr_ids,
                    rew_list,
                    fitness_scores,
                    metrics_dicts,
                    isl_ids,
                    parent_paths_list,            # <--- NEW
                    operator_list 
                )
            else:
                rewards_database.add_individuals_to_islands(
                    gen_ids,
                    ctr_ids,
                    rew_list,
                    fitness_scores,
                    metrics_dicts,
                    isl_ids,
                    parent_paths_list,         # <--- NEW
                    operator_list
                )


            island_info = [
                {
                    island_id: {
                        f"{gen_id}_{count_id}": fitness
                        for gen_id, count_id, fitness in zip(
                            island.generation_ids, island.counter_ids, island.fitness_scores
                        )
                    }
                }
                for island_id, island in enumerate(rewards_database._islands)
            ]
            tracker.log({"generation": generation_id, "islands": island_info})


if __name__ == "__main__":
    main()
