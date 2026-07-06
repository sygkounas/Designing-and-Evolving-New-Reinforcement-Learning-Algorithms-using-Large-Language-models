import os
import random
import sys
from typing import Tuple, List, Dict

import numpy as np
from absl import logging

from evolutionary_utils.entities import Island
from levenshtein import extract_compute_loss_section, normalized_lev as levenshtein_dist




def normalized(x: List[float], temp: float = 0.5):
    x = np.array(x)
    return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)


class RevolveDatabase:
    """
    Adapted from Fun Search: https://github.com/google-deepmind/funsearch/blob/main
    """

    def __init__(
        self,
        num_islands: int,
        max_size: int,
        crossover_prob: float,
        migration_prob: float,
        load_islands: bool,
        reward_fn_dir: str,
        baseline: str,
    ):
        self.reward_fn_dir = reward_fn_dir
        self.num_islands = (
            num_islands  # starting with num_islands, does not increase with crossover
        )
        self.max_size = max_size  # max group size
        self.crossover_prob = crossover_prob
        self.migration_prob = migration_prob
        self.baseline = baseline
        self.heuristic_dir = reward_fn_dir

        self._islands: List[Island] = []
        if load_islands:
            # for it > 0, load stored islands
            for island_id in range(self.num_islands):
                loaded_island = Island.load_island(
                    self.reward_fn_dir, self.baseline, island_id
                )
                self._islands.append(loaded_island)
        else:
            # Initialize empty islands.
            self._islands = [
                Island(island_id, [], [], [], [], [], self.heuristic_dir, self.baseline)
                for island_id in range(self.num_islands)
            ]

    def seed_islands(
        self,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        island_ids: List[int],
        parent_paths_list: List[List[str]],
        operators_list: List[str],
        ):
        """
        for initialization step (generation_id = 0)
        all individuals are added
        """
        for i, (
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            metrics_dict,
            island_id,
        ) in enumerate(
            zip(
                generation_ids,
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )
        ):
            logging.info(
                f"Inside seed_islands: island_id={island_id}, type={type(island_id)}, "
                f"generation_id={generation_id}, counter_id={counter_id}"
            )

            # HERE we finally pass parents + operator for this individual
            self._islands[island_id].register_individual_in_island(
                generation_id,
                counter_id,
                rew_fn_string,
                fitness_score,
                metrics_dict,
                parent_fn_paths=parent_paths_list[i],
                operator=operators_list[i],
            )


    def add_individuals_to_islands(
        self,
        generation_ids: List[int],
        counter_ids: List[int],
        rew_fn_strings: List[str],
        fitness_scores: List[float],
        metrics_dicts: List[dict],
        island_ids: List[int],
        parent_paths_list: List[List[str]],
        operators_list: List[str],
    ):
        for i, (
            generation_id,
            counter_id,
            rew_fn_string,
            fitness_score,
            island_id,
            metrics_dict,
        ) in enumerate(
            zip(
                generation_ids,
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                island_ids,
                metrics_dicts,
            )
        ):
            # corner case: if group is not empty, calculate average fitness score
            if self._islands[island_id].size != 0:
                island_avg_fitness_score = self._islands[
                    island_id
                ].average_fitness_score
            else:
                island_avg_fitness_score = -sys.maxsize - 1
            # for initial generations, add everything
            # check if reward is adding any value to the group
            if fitness_score >= island_avg_fitness_score:
                self._islands[island_id].register_individual_in_island(
                generation_id,
                counter_id,
                rew_fn_string,
                fitness_score,
                metrics_dict,
                parent_fn_paths=parent_paths_list[i],
                operator=operators_list[i],
                )
                logging.info(
                    "Average score of island %d increased to %s",
                    island_id,
                    self._islands[island_id].average_fitness_score,
                )
            else:
                # delete the stored individual txt, models, json
                logging.info(
                    "Fitness score %s for individual lower than average "
                    "Island %d fitness %s, discarding",
                    fitness_score,
                    island_id,
                    island_avg_fitness_score,
                )
                # remove checkpoint and reward history (added during training)
                '''
                reward_history_path = (
                    f"{self.reward_fn_dir}/island_{island_id}/reward_history/"
                    f"{generation_id}_{counter_id}.json"
                )
                model_checkpoint_path = (
                    f"{self.reward_fn_dir}/island_{island_id}/model_checkpoints/"
                    f"{generation_id}_{counter_id}.h5"
                )
                RevolveDatabase.delete_file(
                    reward_history_path, "reward history (.json) file"
                )
                RevolveDatabase.delete_file(
                    model_checkpoint_path, "model checkpoint (.h5) file"
                )
                '''

            # if island size exceeds max size, discard individual with the lowest score
            if self._islands[island_id].size > self.max_size:
                logging.info(
                    "Exceeded maximum size on island %d, "
                    "discarding individual with lowest score",
                    island_id,
                )
                while self._islands[island_id].size > self.max_size:
                    self._islands[island_id].remove_lowest()

        # repeats at the end of each generation
        # reset_prob = (len(self._islands) - self.num_islands) / self.num_islands
     #   if random.random() <= self.migration_prob and len(self._islands) > 1:
       #     self.reset_islands()

    def reset_islands(self):
        """
        Resets the weaker half of islands and seeds them
        with individuals migrated from fitter islands
        """
        print("============ Resetting Island ============")
        # sort best scores after adding minor noise to break ties.
        indices_sorted_by_score = np.argsort(
            np.array([island.best_fitness_score for island in self._islands])
            + np.random.randn(len(self._islands)) * 1e-6
        )
        num_islands_to_reset = len(self._islands) // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for reset_island_id in reset_islands_ids:
            # delete associated files while retaining only the fittest
            self._islands[reset_island_id].only_keep_best()
            # founder island to migrate to the empty island with
            # the size of founder island must be > 1
            founder_island_id = np.random.choice(keep_islands_ids)
            founder_island = self._islands[founder_island_id]
            repeats = 0  # to halt the while loop
            while founder_island.size <= 1:
                founder_island_id = np.random.choice(keep_islands_ids)
                founder_island = self._islands[founder_island_id]
                repeats += 1
                if repeats >= 10:
                    break
            if repeats >= 10:
                # if the while loop has exceeded a certain number of tries, skip
                continue
            # sample an individual from the founder island (NOT the best)
            founder_individual = founder_island.fittest_individual
            while founder_individual == founder_island.fittest_individual:
                founder_individual = random.choices(
                    founder_island.individuals,
                    normalized(founder_island.fitness_scores),
                )[0]
            # register the new (seed) member of the reset island and
            # copy/migrate the relevant files from founder island to the reset_island_id
            logging.info(
                f"Migrating individual from Island {founder_island_id} to Island {reset_island_id}"
            )
            self._islands[reset_island_id].migrate_fn(founder_individual)
            # remove the founder_individual from the founder island
            self._islands[founder_island_id].remove_individual(founder_individual)

    def sample_in_context(
        self, num_samples: Dict, temperature: float
    ) -> Tuple[List[Tuple[str, float]], int, str]:
        """
        New correct sampling logic:
        1) select island
        2) select parent1 (fitness-weighted)
        3) fitness-based micro/macro decision
        4) choose operator (mutation or crossover)
        5) if crossover → select parent2 using fitness + slight diversity (Levenshtein on compute_loss)
        """

        # ----------------------------------------------------
        # STEP 1: Select island (same as before)
        # ----------------------------------------------------
        island_weights = normalized(
            [
                self._islands[island_id].average_fitness_score
                for island_id in range(self.num_islands)
            ],
            temperature,
        )

        sampled_island_id, sampled_island = random.choices(
            list(enumerate(self._islands)), weights=island_weights
        )[0]

        # ----------------------------------------------------
        # STEP 2: Pick first parent (fitness-weighted)
        # ----------------------------------------------------
        parent1_id = np.random.choice(
            range(sampled_island.size),
            p=normalized(sampled_island.fitness_scores, temp=0.5),
        )

        parent1_fn = sampled_island.fn_file_paths[parent1_id]
        parent1_fit = sampled_island.fitness_scores[parent1_id]

        # Save parent1
        in_context_samples = [(parent1_fn, parent1_fit)]

        # ----------------------------------------------------
        # STEP 3: Decide micro vs macro mutation from FITNESS
        # ----------------------------------------------------
        low_lim=0.5
        high_lim=0.75
        if parent1_fit < low_lim:
            p_micro = 0.0
            p_macro = 1.0
        elif parent1_fit > high_lim:
            p_micro = 1.0
            p_macro = 0.0
        else:
            # linear interpolation between 0.35 and 0.80
            t = (parent1_fit - low_lim) / (high_lim - low_lim)
            p_micro = t
            p_macro = 1 - t

        # ----------------------------------------------------
        # STEP 4: Choose operator (65% mut / 35% crossover)
        # ----------------------------------------------------
        if random.random() < 0.65:
            operator = "mutation_micro" if random.random() < p_micro else "mutation_macro"
            logging.info(f"{operator} | sampled island: {sampled_island_id}")
            return in_context_samples, sampled_island_id, operator

        operator = "crossover"

        # ----------------------------------------------------
        # STEP 5: Select second parent for CROSSOVER
        # ----------------------------------------------------
        if sampled_island.size < 2:
            # fallback if island too small
            operator = "mutation_micro" if random.random() < p_micro else "mutation_macro"
            logging.info(f"{operator} | sampled island: {sampled_island_id}")
            return in_context_samples, sampled_island_id, operator

        # Load compute_loss of parent1
        with open(parent1_fn, "r") as f:
            parent1_core = extract_compute_loss_section(f.read())

        # Score = 0.9 * fitness + 0.1 * levenshtein_diversity
        scores = []
        for idx, fn in enumerate(sampled_island.fn_file_paths):
            

            with open(fn, "r") as f:
                code2 = f.read()
            p2_core = extract_compute_loss_section(code2)

            lev = levenshtein_dist(parent1_core, p2_core)
            fit = sampled_island.fitness_scores[idx]
            alpha=0.5

            score = alpha * fit + (1 - alpha) * lev
            scores.append(score)

        # Normalize scores → probs
        probs = normalized(scores, temp=temperature)
        parent2_id = np.random.choice(range(sampled_island.size), p=probs)


        parent2_fn = sampled_island.fn_file_paths[parent2_id]
        parent2_fit = sampled_island.fitness_scores[parent2_id]

        in_context_samples.append((parent2_fn, parent2_fit))

        logging.info(f"Crossover | sampled island: {sampled_island_id}")
        return in_context_samples, sampled_island_id, operator

    @staticmethod
    def delete_file(filepath: str, filetype: str):
        if os.path.exists(filepath):
            logging.info(f"Removing {filetype} from {filepath}.")
            os.remove(filepath)
        else:
            logging.info(f"{filetype} does not exist in {filepath}.")

