import os
import glob
import numpy as np
from rapidfuzz.distance import Levenshtein
import re

RUNS_DIR = "/home/alkis/Downloads/new_rl_stuff/runs"


# ---------------------------------------------------------
# Extract only the algorithm's core (compute_loss section)
# ---------------------------------------------------------

def extract_compute_loss_section(code_text: str) -> str:
    """
    Extract only the compute_loss(...) block from the algorithm file.
    This is the algorithm's 'brain' and is what should be compared.
    """
    pattern = r"def compute_loss\(.*?\):(.*?)(?=\n\s*def )"
    m = re.search(pattern, code_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return code_text  # fallback: use whole file


# ---------------------------------------------------------
# Load algorithm files from algo_*/code/
# ---------------------------------------------------------

def load_algorithms(run_dir=RUNS_DIR):
    """
    Loads algorithm .py files from algo_*/code/.
    Returns dict {algo_name: core_algorithm_text}
    (core = compute_loss section only)
    """
    algos = {}

    # find all directories like algo_0, algo_1, ...
    for algo_dir in sorted(glob.glob(os.path.join(run_dir, "algo_*"))):
        code_dir = os.path.join(algo_dir, "code")

        if not os.path.isdir(code_dir):
            continue

        # find python files inside /code/
        py_files = glob.glob(os.path.join(code_dir, "*.py"))
        if len(py_files) == 0:
            continue

        # priority: use refine_final.py if present
        final = [f for f in py_files if os.path.basename(f) == "refine_final.py"]
        if final:
            chosen = final[0]
        else:
            chosen = sorted(py_files)[-1]   # fallback = newest refinement

        with open(chosen, "r", encoding="utf-8") as f:
            full_code = f.read()

        core = extract_compute_loss_section(full_code)
        algos[os.path.basename(algo_dir)] = core

    return algos


# ---------------------------------------------------------
# Diversity on the extracted compute-loss blocks
# ---------------------------------------------------------

def normalized_lev(a, b):
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def compute_population_diversity(algos_dict):
    names = list(algos_dict.keys())
    texts = [algos_dict[k] for k in names]
    n = len(texts)

    if n <= 1:
        return 0.0, np.zeros((n, n)), names

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = normalized_lev(texts[i], texts[j])
            mat[i, j] = mat[j, i] = d

    mean_div = mat[np.triu_indices(n, k=1)].mean()
    return mean_div, mat, names


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    algos = load_algorithms()
    mean_div, mat, names = compute_population_diversity(algos)

    print("Algorithms loaded:", names)
    print("Mean diversity (compute_loss only):", round(mean_div, 4))
    print("Matrix:\n", mat)
