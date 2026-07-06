"""
Evolutionary Utilities
"""

import glob
import json
import os
import sys
from typing import List, Optional
import shutil
import re
from rapidfuzz.distance import Levenshtein

import numpy as np
from absl import logging


def _extract_compute_loss_section(code_text: str) -> str:
    """
    Extract only the compute_loss(...) body, same logic as levenshtein.py.
    """
    pattern = r"def compute_loss\(.*?\):(.*?)(?=\n\s*def )"
    m = re.search(pattern, code_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return code_text.strip()


def _normalized_lev(a: str, b: str) -> float:
    """
    Normalized Levenshtein distance on text strings.
    """
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)



class Individual:
    """Single Individual in an Island"""

    def __init__(
        self,
        island_id: int,
        generation_id: int,
        counter_id: int,
        rew_fn_string: str,
        fitness_score: float,
        metrics_dict: dict,  # other performance metrics
        reward_fn_dir: str,
        parents=None,
        operator: str = "seed",
    ):
        self.island_id = island_id
        self.generation_id = generation_id
        self.counter_id = counter_id
        self.rew_fn_string = rew_fn_string
        self.fitness_score = fitness_score
        self.reward_history = None
        self.model_checkpoint = None
        self.metrics_dict = metrics_dict
        self.reward_fn_dir = reward_fn_dir
        self.parents = parents or []          # list[str] of parent fn paths
        self.operator = operator              # "seed", "mutation_micro", "crossover", ...

    @property
    def fn_file_path(self):
        """function string path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/generated_fns/"
            f"{self.generation_id}_{self.counter_id}.txt"
        )

    @property
    def fitness_file_path(self):
        """fitness score file path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/fitness_scores/"
            f"{self.generation_id}_{self.counter_id}.txt"
        )

    @property
    def reward_history_path(self):
        """reward history file path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/reward_history/"
            f"{self.generation_id}_{self.counter_id}.json"
        )

    @property
    def model_checkpoint_path(self):
        """model (policy) checkpoint path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/model_checkpoints/"
            f"{self.generation_id}_{self.counter_id}.h5"
        )

    @property
    def human_feedback_path(self):
        """human feedback path"""
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/human_feedback/"
            f"{self.generation_id}_{self.counter_id}.h5"
        )
    
    @property
    def parents_entry_dir(self) -> str:
        return (
            f"{self.reward_fn_dir}/island_{self.island_id}/parents/"
            f"{self.generation_id}_{self.counter_id}"
        )

    @property
    def parents_info_path(self) -> str:
        return os.path.join(self.parents_entry_dir, "info.json")

    def save_parents(self, parent_fn_paths: List[str], operator: str):
        """
        Save parent copies + lineage meta, including Levenshtein distance
        between parent(s) and this child on the compute_loss section.
        """
        if not parent_fn_paths:
            return

        os.makedirs(self.parents_entry_dir, exist_ok=True)

        copied = []
        parent_codes = []

        # Copy parents and load their code
        for src in parent_fn_paths:
            parent_name = os.path.basename(src)   # e.g. "0_3.txt"
            dst = os.path.join(self.parents_entry_dir, parent_name)

            # load parent code
            with open(src, "r") as fsrc:
                code_parent = fsrc.read()
            parent_codes.append(code_parent)

            # save as real filename
            with open(dst, "w") as fdst:
                fdst.write(code_parent)

            copied.append(parent_name)


        # --- Levenshtein on compute_loss sections ---
        child_core = _extract_compute_loss_section(self.rew_fn_string)
        lev_per_parent = []
        for code_parent in parent_codes:
            parent_core = _extract_compute_loss_section(code_parent)
            lev_per_parent.append(_normalized_lev(parent_core, child_core))

        if lev_per_parent:
            lev_avg = float(sum(lev_per_parent) / len(lev_per_parent))
        else:
            lev_avg = 0.0

        info = {
            "operator": operator,
            "num_parents": len(parent_fn_paths),
            "parent_src_files": copied,
            "levenshtein_to_child_per_parent": lev_per_parent,
            "levenshtein_to_child_avg": lev_avg,
        }

        with open(self.parents_info_path, "w") as f:
            json.dump(info, f)



    def save_files(self):
        """
        saves fn string, scalar fitness, metrics dict, parents
        """
        base_path = f"{self.reward_fn_dir}/island_{self.island_id}"

        # Ensure directories exist
        for dir_name in [
            "generated_fns",
            "fitness_scores",
            "reward_history",
            "model_checkpoints",
            "parents",
        ]:
            os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)

        # -------------------------
        # 1) Save reward fn .txt
        # -------------------------
        with open(self.fn_file_path, "w") as f:
            f.write(self.rew_fn_string)

        # -------------------------
        # 2) Save FITNESS (scalar)
        #    fitness_scores/<gen>_<counter>.txt
        # -------------------------
        with open(self.fitness_file_path, "w") as f:
            f.write(f"{self.fitness_score}\n")

        # -------------------------
        # 3) Save METRICS (large JSON dict)
        #    reward_history/<gen>_<counter>.json
        # -------------------------
        metrics_path = os.path.join(
            base_path, "reward_history", f"{self.generation_id}_{self.counter_id}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_dict, f, indent=2)

        # -------------------------
        # 4) Save PARENTS
        #    parents/<gen>_<counter>.json
        # -------------------------
        parents_dir = os.path.join(base_path, "parents", f"{self.generation_id}_{self.counter_id}")
        os.makedirs(parents_dir, exist_ok=True)

        info_path = os.path.join(parents_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump({
                "operator": self.operator,
                "num_parents": len(self.parents),
                "parent_src_files": [os.path.basename(p) for p in self.parents]
            }, f, indent=2)
            


    def remove_files(self):
        """
        delete associated files from the database for the individual
        deletes generated_fn txt file, fitness score, reward history, model checkpoint file
        fn_path: root_path/database/island_{island_id}/generated_fns/gen_counter.txt
        """

        def delete_file(filepath: str, filetype: str):
            if os.path.exists(filepath):
                logging.info(f"Removing {filetype} from {filepath}.")
                os.remove(filepath)
            else:
                logging.info(f"{filetype} does not exist in {filepath}.")

        delete_file(self.fn_file_path, "generated reward fn (.txt) file")
        delete_file(self.fitness_file_path, "fitness score (.txt) file")
        delete_file(self.reward_history_path, "reward history (.json) file")
        delete_file(self.model_checkpoint_path, "model checkpoint (.h5) file")
        if os.path.isdir(self.parents_entry_dir):
            logging.info(f"Removing parents dir from {self.parents_entry_dir}.")
            shutil.rmtree(self.parents_entry_dir)



class Island:
    """A population of individuals: aka island"""

    def __init__(
        self,
        island_id: int,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        reward_fn_dir: str,
        baseline: str,
    ):
        self.reward_fn_dir = reward_fn_dir
        self.island_id = island_id
        
        self.individuals = [
            Individual(
                self.island_id,
                generation_id,
                counter_id,
                rew_fn_str,
                fitness_score,
                metrics_dict,
                self.reward_fn_dir,
            )
            for generation_id, counter_id, rew_fn_str, fitness_score, metrics_dict in zip(
                generation_ids,
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
            )
        ]
        self.baseline = baseline

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def fitness_scores(self) -> List[float]:
        if self.size == 0:
            return [-sys.maxsize - 1]
        return [individual.fitness_score for individual in self.individuals]

    @property
    def best_fitness_score(self) -> float:
        return max(self.fitness_scores)

    @property
    def average_fitness_score(self):
        return np.mean(self.fitness_scores)

    @property
    def fittest_individual(self) -> Individual:
        fittest_ind = np.argmax(
            [individual.fitness_score for individual in self.individuals]
        )
        return self.individuals[fittest_ind]

    @property
    def generation_ids(self) -> List[int]:
        return [individual.generation_id for individual in self.individuals]

    @property
    def counter_ids(self) -> List[int]:
        return [individual.counter_id for individual in self.individuals]

    @property
    def fn_file_paths(self) -> List[str]:
        return [individual.fn_file_path for individual in self.individuals]

    @property
    def reward_history_paths(self) -> List[str]:
        return [individual.reward_history_path for individual in self.individuals]



    def register_individual_in_island(
        self,
        generation_id: int,
        counter_id: int,
        rew_fn_string: str,
        fitness_score: float,
        metrics_dict: dict,
        parent_fn_paths: Optional[List[str]] = None,
        operator: str = "seed",
    ):
        """
        add individual to the island population.
        """

        logging.info(
            f"Registering Individual with generation_id {generation_id}"
            f" | counter_id {counter_id} in Island {self.island_id}."
        )
        new_individual = Individual(
            self.island_id,
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            metrics_dict,
            self.reward_fn_dir,
            parents=parent_fn_paths or [],
            operator=operator,
        )
        self.individuals.append(new_individual)
        new_individual.save_files()
        new_individual.save_parents(parent_fn_paths or [], operator)


    @classmethod
    def load_island(cls, heuristic_dir: str, baseline: str, island_id: int = 0):
        base_dir = os.path.join(f"{heuristic_dir}/island_{island_id}")
        all_fns_file_paths = sorted(glob.glob(f"{base_dir}/fitness_scores/*.txt"))

        fitness_scores = []
        all_generation_ids = []
        all_counter_ids = []
        all_fn_strings = []
        all_metrics = []

        # temp store of loaded parent info keyed by (gen, ctr)
        parent_info = {}

        for fn_path in all_fns_file_paths:
            filename = os.path.basename(fn_path)  # "gen_ctr.txt"
            gen_str, ctr_str = os.path.splitext(filename)[0].split("_")
            generation_id = int(gen_str)
            counter_id = int(ctr_str)

            fitness_score_filepath = os.path.join(base_dir, "fitness_scores", filename)
            with open(fitness_score_filepath, "r") as f:
                fitness_value = float(f.read().strip())
            fitness_scores.append(fitness_value)
            metrics_filepath = os.path.join(base_dir, "reward_history", filename.replace(".txt", ".json"))
            with open(metrics_filepath, "r") as f:
                metrics_dict = json.load(f)
            all_metrics.append(metrics_dict)
            all_generation_ids.append(generation_id)
            all_counter_ids.append(counter_id)

            with open(fn_path, "r") as f:
                all_fn_strings.append(f.read())

            # load parents metadata if exists
            pdir = os.path.join(base_dir, "parents", f"{generation_id}_{counter_id}")
            info_path = os.path.join(pdir, "info.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    parent_info[(generation_id, counter_id)] = json.loads(f.read())
            else:
                parent_info[(generation_id, counter_id)] = {"operator": None, "num_parents": 0}

        island = cls(
            island_id,
            all_generation_ids,
            all_counter_ids,
            all_fn_strings,
            fitness_scores,
            all_metrics,
            heuristic_dir,
            baseline,
        )

        # attach lineage metadata to individual objects (no metrics changes)
        for ind in island.individuals:
            info = parent_info.get((ind.generation_id, ind.counter_id), {"operator": None, "num_parents": 0})
            ind.lineage_operator = info.get("operator")
            ind.num_parents = info.get("num_parents", 0)

            # also store parent IDs parsed from copied filenames (if present)
            ind.parent_ids = []
            for name in info.get("parent_src_files", []) or []:
                # expects "gen_ctr.txt"
                base = os.path.splitext(name)[0]
                if "_" in base:
                    try:
                        g, c = base.split("_")
                        ind.parent_ids.append((int(g), int(c)))
                    except Exception:
                        pass

        return island


    def remove_lowest(self):
        """
        removes the individual with the lowest fitness score in the island
        """

        lowest_score_index = np.argmin(
            [individual.fitness_score for individual in self.individuals]
        )
        weakest_individual = self.individuals.pop(lowest_score_index)
        weakest_individual.remove_files()

    def remove_individual(self, to_remove_individual: Individual):
        """
        remove an individual from the island
        """
        for individual_id, individual in enumerate(self.individuals):
            if (
                individual.generation_id == to_remove_individual.generation_id
                and individual.counter_id == to_remove_individual.counter_id
            ):
                self.individuals.pop(individual_id)
                individual.remove_files()
                return

    def only_keep_best(self):
        """
        remove all except best individual
        """
        fittest_individual = self.fittest_individual
        # TODO: handle case where more than one are equally fit
        for individual in self.individuals:
            if individual.fitness_score == fittest_individual.fitness_score:
                continue
            individual.remove_files()
        self.individuals = [fittest_individual]

    def migrate_fn(
        self,
        founder_individual: Individual,
    ):
        """
        migrate individual from founder island to current island
        """

        # register founder fn in the reset island
        self.register_individual_in_island(
            founder_individual.generation_id,
            founder_individual.counter_id,
            founder_individual.rew_fn_string,
            founder_individual.fitness_score,
            founder_individual.metrics_dict,
        )
