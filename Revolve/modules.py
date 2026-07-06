"""
Various stages of individual generation, training, and evaluation:
1. Reward Function Generation
2. Policy Training
3. Policy Evaluation
"""

import concurrent.futures
import json
import multiprocessing
import os
import time
from typing import Tuple, Optional, Dict

import absl.logging as logging
import hydra
import openai
from hydra.core.global_hydra import GlobalHydra
from openai import OpenAI
from typing import List, Tuple, Dict, Optional

openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)




import re
from pathlib import Path

MODEL = "gpt-5.2"   # or whatever you use


def call_model(client, prompt: str, save_dir: Path):
   # save_dir.mkdir(exist_ok=True, parents=True)
    #inp = save_dir / "llm_input.txt"
    #out = save_dir / "llm_output.txt"

    #inp.write_text(prompt, encoding="utf-8")
    resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    reasoning_effort="high",   # or "default" / "low"
)
    text = resp.choices[0].message.content
   # out.write_text(text, encoding="utf-8")
    return text


def extract_algo_block(text: str) -> str:
    match = re.search(
        r"(?:```python|python\s*''')\s*(.*?)\s*(?:```|''')",
        text,
        re.DOTALL
    )
    if not match:
        raise ValueError("No python code block found in LLM output.")
    return match.group(1).strip()


# generates reward functions
class RewardFunctionGeneration:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.llm = "gpt-5.1"  # use the new model

    def query_llm(self, in_context_prompt: str) -> Tuple[str, int, int]:
        # New API: no temperature, no top_p, no penalties
        resp = client.chat.completions.create(
            model=self.llm,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt 
                },
                {
                    "role": "user",
                    "content": in_context_prompt,
                },
            ],
        )

        # Extract content + token counts
        out = resp.choices[0].message.content
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        return out, prompt_tokens, completion_tokens

    @staticmethod
    def prepare_in_context_prompt(
            in_context_samples: Optional[List[Tuple[str, float]]],
            operator_prompt: str,
            evolve: bool,
            baseline: str,
    ) -> str:
        """
        Replace placeholders {Your_Algorithm}, {Metrics}, {Fitness}
        with actual content from parents.
        Works for mutation (1 parent) and crossover (2 parents).
        """

        if not evolve or not in_context_samples:
            return operator_prompt

        # If multiple parents (crossover), concatenate blocks
        final_algorithm_block = ""
        final_metrics_block = ""
        final_fitness_block = ""

        for filename, scalar_fitness in in_context_samples:
            # -------------------------
            # 1) Algorithm code
            # -------------------------
            algo_code = open(filename, "r").read()

            # -------------------------
            # 2) Metrics JSON
            # -------------------------
            metrics_file = filename.replace("generated_fns", "reward_history") \
                                .replace(".txt", ".json")
            with open(metrics_file, "r") as f:
                metrics_json = f.read().strip()

            # -------------------------
            # 3) Summarized fitness JSON
            # -------------------------
            base_dir = os.path.dirname(os.path.dirname(filename))  # .../island_X
            g_c = os.path.splitext(os.path.basename(filename))[0]   # "g_c"
            fitness_summary_path = os.path.join(
                base_dir, "rl_logs", g_c, "training_final", "final_training.json"
            )

            if os.path.exists(fitness_summary_path):
                with open(fitness_summary_path, "r") as f:
                    fitness_summary_json = f.read().strip()
            else:
                fitness_summary_json = "{}"

            # accumulate if multiple parents
            final_algorithm_block += f"\n\n{algo_code}\n\n"
            final_metrics_block += f"\n\n{metrics_json}\n\n"
            final_fitness_block += f"\n\n{fitness_summary_json}\n\n"

        operator_prompt = operator_prompt.replace("{Your_Algorithm}", final_algorithm_block)
       # operator_prompt = operator_prompt.replace("{Metrics}", final_metrics_block)
        operator_prompt = operator_prompt.replace("{Fitness}", final_fitness_block)
       # print("Prepared in-context prompt with parent algorithms.",operator_prompt)

        return operator_prompt




    def generate_rf(self, in_context_prompt: str):
        from pathlib import Path

        save_dir = Path("llm_calls")   # can be any folder you want

        raw_output = call_model(client, in_context_prompt, save_dir)
        python_code = extract_algo_block(raw_output)

        return python_code


