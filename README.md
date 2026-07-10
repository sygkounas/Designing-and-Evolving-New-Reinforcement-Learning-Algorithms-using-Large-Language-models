# Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.28416-b31b1b.svg)](https://arxiv.org/abs/2603.28416)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](media/Designing_and_Evolving_New_Reinforcement_Learning_Algorithms_using_Large_Language_models%20%284%29.pdf)
[![Conference](https://img.shields.io/badge/GECCO-2026-1f6feb.svg)](https://doi.org/10.1145/3795095.3805180)

Official repository for **“Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models”** by Alkis Sygkounas, Amy Loutfi, and Andreas Persson.

Instead of tuning a fixed reinforcement-learning algorithm, this work searches directly over **executable update rules**. Each candidate is a complete training procedure represented as Python code. Large language models act as generative variation operators, while full RL training runs provide the fitness signal.

The framework extends [REvolve (ICLR 2025)](https://openreview.net/forum?id=cJPUpL8mOw) from reward-function evolution to reinforcement-learning algorithm discovery.

## Framework overview

[View the framework overview](media/framework_overview_rl.pdf)

The search keeps the policy architecture, optimizer, rollout procedure, and training budget fixed. Evolution therefore changes the **learning logic**, rather than network capacity or optimizer choice.

### LLM-guided variation

Two operators generate new executable update rules:

- **Macro mutation** rewrites one semantically coherent mechanism, such as an auxiliary objective, planning component, or stabilization rule.
- **Diversity-aware crossover** combines mechanisms from two parent algorithms selected within the same island.

Canonical mechanisms are prohibited during evolution, including actor–critic decomposition, temporal-difference updates, and value bootstrapping. This discourages the LLM from simply reproducing PPO, DQN, or SAC.

### Diversity-aware parent selection

Parent 1 is sampled from the current island according to fitness. Every individual in the same island is then considered as a possible second parent using

\[
S(f_2\mid f_1)=\alpha F(f_2)+(1-\alpha)d_{\mathrm{lev}}(f_1,f_2),
\]

where:

- \(F(f_2)\) is the fitness of the candidate second parent;
- \(d_{\mathrm{lev}}(f_1,f_2)\) is the normalized Levenshtein distance between the source code of the two `compute_loss` functions;
- \(\alpha\) controls the trade-off between performance and structural diversity.

A softmax over these scores defines the probability of sampling each candidate as parent 2. This reduces crossover between near-duplicate algorithms while still favouring high-performing parents.

### Fitness

Each candidate is evaluated by complete RL training on five environments:

- CartPole-v1
- LunarLander-v3
- MountainCar-v0
- Acrobot-v1
- HalfCheetah-v5

Each environment is trained with five independent random seeds, giving **25 training runs per candidate**. For each seed, the maximum evaluation return reached during training is retained. The seed-level maxima are averaged, normalized using fixed environment-specific bounds, and then averaged across environments to obtain one scalar fitness.

A candidate is accepted when its fitness is at least the mean fitness of its island, replacing the island’s lowest-fitness individual.

### Post-evolution refinement

RL update rules can be highly sensitive to their internal scalar coefficients. After evolution, the best update-rule structures are therefore refined using **LLM-guided hyperparameter optimization (LLM-HPO)**:

1. The LLM reads the complete update-rule code and environment specifications.
2. It proposes a bounded numeric interval for each internal scalar parameter.
3. Parameter vectors are sampled uniformly from the resulting search region.
4. Each sampled configuration is evaluated with the same training protocol.
5. The best-performing configuration is retained for final evaluation.

The update-rule structure remains fixed during this stage; only its scalar parameters are refined.

## Evolved algorithms

### CG-FPD

**Confidence-Guided Forward Policy Distillation** learns a compact latent world model, generates short action sequences, and predicts their outcomes. Sequences are preferred when they produce higher predicted reward and avoid termination. Their first actions are combined into a training target for the policy, so execution requires only a direct policy forward pass.

### DF-CWP-CP

**Differentiable Forward Confidence-Weighted Planning with Controllability Prior** learns an observation-space world model together with confidence estimates for its state, reward, and termination predictions. It improves the policy by backpropagating through short imagined rollouts, down-weighting unreliable predictions and adding a controllability objective based on action sensitivity.

Both algorithms are **critic-free** and **planning-driven**: they do not use value functions, TD targets, Bellman backups, or policy-gradient estimators.

## Results

The two selected algorithms are evaluated on ten Gymnasium environments. Five environments are used during evolution and five are held out from evolutionary fitness to assess generalization.

### Evolutionary search

- [Maximum population fitness: GPT-5.2 vs. Claude 4.5 Opus](media/max_fitness_gpt_vs_claude.pdf)
- [Ablation of the Levenshtein weight](media/ablation_alpha_0_vs_1.pdf)

### Environments used during evolution

| Environment | Result |
|---|---|
| CartPole-v1 | [PDF](media/CartPole-v1.pdf) |
| LunarLander-v3 | [PDF](media/LunarLander-v3.pdf) |
| MountainCar-v0 | [PDF](media/MountainCar-v0.pdf) |
| Acrobot-v1 | [PDF](media/Acrobot-v1.pdf) |
| HalfCheetah-v5 | [PDF](media/HalfCheetah-v5.pdf) |

### Environments unseen during evolution

| Environment | Result |
|---|---|
| Walker2d-v5 | [PDF](media/Walker2d-v5.pdf) |
| InvertedPendulum-v5 | [PDF](media/InvertedPendulum-v5.pdf) |
| Reacher-v5 | [PDF](media/Reacher-v5.pdf) |
| Swimmer-v5 | [PDF](media/Swimmer-v5.pdf) |
| Pusher-v5 | [PDF](media/Pusher-v5.pdf) |

The evolved algorithms achieve competitive performance against PPO, A2C, DQN, and SAC across the ten-environment benchmark suite, including the five environments not used during evolution.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/sygkounas/evolutionary_discovery.git
cd evolutionary_discovery

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The exact package versions used for the experiments will be pinned in `requirements.txt`.

## Paper

- [Local PDF](media/Designing_and_Evolving_New_Reinforcement_Learning_Algorithms_using_Large_Language_models%20%284%29.pdf)
- [arXiv](https://arxiv.org/abs/2603.28416)

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{sygkounas2026evolutionary,
  title     = {Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models},
  author    = {Sygkounas, Alkis and Loutfi, Amy and Persson, Andreas},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)},
  year      = {2026},
  doi       = {10.1145/3795095.3805180}
}
```

This work builds on REvolve:

```bibtex
@inproceedings{hazra2025revolve,
  title     = {{RE}volve: Reward Evolution with Large Language Models using Human Feedback},
  author    = {Rishi Hazra and Alkis Sygkounas and Andreas Persson and Amy Loutfi and Pedro Zuidberg Dos Martires},
  year      = {2025},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  url       = {https://openreview.net/forum?id=cJPUpL8mOw}
}
```

## Acknowledgements

This work is supported by the Knut and Alice Wallenberg Foundation through the Wallenberg AI, Autonomous Systems and Software Program and the Wallenberg Scholars Grant. Computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden through the LUMI supercomputer.Code and videos coming soon...
