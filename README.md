# Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models

[![Paper](https://img.shields.io/badge/Paper-PDF-355C8A.svg)](media/Designing_and_Evolving_New_Reinforcement_Learning_Algorithms_using_Large_Language_models%20%284%29.pdf)
[![GECCO 2026](https://img.shields.io/badge/GECCO-2026-243B64.svg)](https://doi.org/10.1145/3795095.3805180)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

Official repository for:

> **Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models**  
> Alkis Sygkounas, Amy Loutfi, and Andreas Persson  
> Genetic and Evolutionary Computation Conference, GECCO 2026

This work searches directly over **executable reinforcement-learning update rules**. Each candidate is a complete training procedure represented as Python code. Large language models generate structural variations, while complete reinforcement-learning training runs provide the fitness signal.

The framework extends [REvolve, ICLR 2025](https://openreview.net/forum?id=cJPUpL8mOw), from evolving reward functions to evolving the reinforcement-learning algorithm itself.

---

## Framework

[![Framework overview](media/framework_overview_rl-1.png)](media/framework_overview_rl.pdf)

The policy architecture, optimizer, rollout procedure, and training budget are fixed. Evolution therefore changes the **learning logic**, rather than network capacity or optimizer choice.

Each candidate update rule is evaluated through complete reinforcement-learning training and receives one aggregated fitness score.

### LLM-guided variation

The language model acts as a generative variation operator through:

- **Macro mutation**, which rewrites one coherent mechanism of the update rule.
- **Diversity-aware crossover**, which combines mechanisms from two parent update rules.

During evolutionary generation, canonical mechanisms are prohibited, including actor–critic decomposition, temporal-difference updates, and value bootstrapping. This discourages the language model from simply reproducing PPO, DQN, or SAC.

### Diversity-aware crossover

Both parents are selected from the same island.

Parent 1 is sampled according to fitness. Every existing individual in that island is then considered as a possible second parent using:

\[
S(f_2\mid f_1)
=
\alpha F(f_2)
+
(1-\alpha)d_{\mathrm{lev}}(f_1,f_2),
\]

where:

- \(F(f_2)\) is the fitness of the possible second parent.
- \(d_{\mathrm{lev}}(f_1,f_2)\) is the normalized Levenshtein distance between the source code of the two `compute_loss` functions.
- \(\alpha\) controls the trade-off between fitness and structural dissimilarity.

A softmax over these scores defines the probability of selecting each individual as parent 2. This reduces crossover between near-duplicate algorithms while retaining pressure toward high-performing parents.

### Evolutionary fitness

Each candidate is trained on five environments:

- CartPole-v1
- LunarLander-v3
- MountainCar-v0
- Acrobot-v1
- HalfCheetah-v5

Each environment is evaluated using five independent random seeds, resulting in:

\[
5\ \text{environments}
\times
5\ \text{seeds}
=
25\ \text{training runs per candidate}.
\]

For each environment:

1. Training is periodically evaluated.
2. The maximum evaluation return reached by each seed is retained.
3. The five seed-level maxima are averaged.
4. The result is normalized using fixed environment-specific bounds.

The final fitness is the mean normalized score across the five environments.

A new candidate is accepted when its fitness is at least the mean fitness of its island. Accepted candidates replace the island’s lowest-fitness individuals.

---

## Post-evolution refinement

Reinforcement-learning update rules are highly sensitive to internal scalar parameters. A strong update-rule structure may therefore appear weak when evaluated using only one parameterization.

After evolution, the selected update rules undergo **LLM-guided hyperparameter optimization**, or **LLM-HPO**:

1. The language model reads the complete update-rule code and the environment specifications.
2. It proposes a bounded numeric interval for each internal scalar parameter.
3. Hyperparameter configurations are sampled uniformly from the resulting search region.
4. Each configuration is evaluated using the same multi-environment fitness protocol.
5. The highest-performing configuration is retained for final evaluation.

The update-rule structure remains fixed during this stage. Only its internal scalar coefficients are refined.

---

## Evolved algorithms

### CG-FPD

**Confidence-Guided Forward Policy Distillation**

CG-FPD learns a compact latent world model of the environment. It performs short-horizon model-based planning by generating candidate action sequences and predicting their latent-state, reward, and termination outcomes.

Action sequences predicted to obtain higher reward and avoid termination receive greater weight. For continuous control, the **Cross-Entropy Method** repeatedly shifts the action-sampling distribution toward higher-scoring sequences.

The first actions of these sequences are combined into a training target for the policy. This implements **planning-to-policy distillation**: planning is used to train the policy, while execution requires only a direct policy forward pass.

### DF-CWP-CP

**Differentiable Forward Confidence-Weighted Planning with Controllability Prior**

DF-CWP-CP learns an observation-space world model that predicts state changes, rewards, and termination probabilities. Separate confidence models estimate the reliability of these predictions.

The current policy is rolled forward through three imagined transitions. The confidence-weighted reward and termination objective is then backpropagated through the world model directly into the policy.

The algorithm additionally optimizes a controllability objective:

\[
\left\|
\frac{\partial\,\text{score}}
{\partial\,\text{action}}
\right\|,
\]

which measures how strongly the predicted planning score changes when the action changes. This provides an additional learning signal when immediate rewards provide little distinction between actions.

Both evolved algorithms are:

- critic-free;
- planning-driven;
- free of temporal-difference targets;
- free of Bellman value bootstrapping;
- free of conventional policy-gradient estimators.

---

## Evolutionary results

### Maximum population fitness

[![Maximum population fitness for GPT-5.2 and Claude 4.5 Opus](media/max_fitness_gpt_vs_claude-1.png)](media/max_fitness_gpt_vs_claude.pdf)

The figure shows the maximum population fitness across ten generations. GPT-5.2 consistently produced higher-fitness update rules than Claude 4.5 Opus across the independent evolutionary runs.

### Diversity-weight ablation

[![Levenshtein diversity-weight ablation](media/ablation_alpha_0_vs_1.png)](media/ablation_alpha_0_vs_1.pdf)

The crossover weight \(\alpha\) controls the balance between second-parent fitness and source-code dissimilarity.

The main experiments use:

\[
\alpha = 0.5.
\]

This intermediate setting achieved higher final fitness than either extreme, indicating that effective crossover requires both functional quality and structural diversity.

> The current repository contains the ablation figure as a PDF. A PNG preview can also be added later if inline rendering is desired.

---

## Final benchmark results

All reported metrics are episodic returns, so **higher is better**, including environments with negative-valued returns.

Results show mean ± standard deviation across five random seeds. For each seed, the best checkpoint reached during training is evaluated over 100 episodes.

| Environment | PPO | A2C | DQN | SAC | CG-FPD | DF-CWP-CP |
|---|---:|---:|---:|---:|---:|---:|
| CartPole ↑ | **500.0 ± 0.0** | **500.0 ± 0.0** | **500.0 ± 0.0** | — | **500.0 ± 0.0** | **500.0 ± 0.0** |
| LunarLander ↑ | 246.6 ± 30.9 | 246.6 ± 13.4 | 250.1 ± 4.1 | — | 241.2 ± 11.0 | **260.6 ± 19.1** |
| MountainCar ↑ | −128.1 ± 44.6 | −134.2 ± 8.7 | −147.8 ± 40.4 | — | **−105.8 ± 10.5** | −108.7 ± 7.3 |
| Acrobot ↑ | −63.5 ± 0.8 | −213.6 ± 202.5 | **−61.9 ± 0.1** | — | −90.6 ± 5.9 | −78.7 ± 8.7 |
| HalfCheetah ↑ | 1579 ± 644 | 796 ± 148 | — | **4989 ± 2668** | 2408 ± 312 | 2104 ± 247 |
| Reacher ↑ | −3.15 ± 0.14 | −6.48 ± 0.57 | — | **−2.05 ± 0.19** | −2.67 ± 0.15 | −5.43 ± 0.44 |
| Swimmer ↑ | 95.0 ± 29.2 | 49.1 ± 1.1 | — | 87.8 ± 23.5 | **247.5 ± 35.5** | 219.8 ± 32.3 |
| InvertedPendulum ↑ | **1000 ± 0** | **1000 ± 0** | — | **1000 ± 0** | **1000 ± 0** | **1000 ± 0** |
| Walker2d ↑ | 3163 ± 397 | 802 ± 291 | — | **4595 ± 252** | 1604 ± 147 | 1298 ± 63 |
| Pusher ↑ | **−25.5 ± 0.3** | −32.4 ± 0.8 | — | **−25.5 ± 0.3** | −27.2 ± 0.9 | −39.9 ± 0.9 |

A dash indicates that the algorithm is not applicable to that action-space type.

The evolved algorithms achieve competitive performance on the complete ten-environment benchmark suite and obtain the strongest reported performance on LunarLander, MountainCar, and Swimmer.

---

## Per-environment learning curves

The first five environments were used to calculate evolutionary fitness. The remaining five were unseen during evolution and were used to assess generalization.

<table>
<tr>
<td align="center" width="50%">
<a href="media/CartPole-v1.pdf">
<img src="media/CartPole-v1-1.png" alt="CartPole-v1 results" width="100%">
</a>
<br>
<strong>CartPole-v1</strong><br>
Used during evolution
</td>

<td align="center" width="50%">
<a href="media/LunarLander-v3.pdf">
<img src="media/LunarLander-v3-1.png" alt="LunarLander-v3 results" width="100%">
</a>
<br>
<strong>LunarLander-v3</strong><br>
Used during evolution
</td>
</tr>

<tr>
<td align="center">
<a href="media/MountainCar-v0.pdf">
<img src="media/MountainCar-v0-1.png" alt="MountainCar-v0 results" width="100%">
</a>
<br>
<strong>MountainCar-v0</strong><br>
Used during evolution
</td>

<td align="center">
<a href="media/Acrobot-v1.pdf">
<img src="media/Acrobot-v1-1.png" alt="Acrobot-v1 results" width="100%">
</a>
<br>
<strong>Acrobot-v1</strong><br>
Used during evolution
</td>
</tr>

<tr>
<td align="center">
<a href="media/HalfCheetah-v5.pdf">
<img src="media/HalfCheetah-v5-1.png" alt="HalfCheetah-v5 results" width="100%">
</a>
<br>
<strong>HalfCheetah-v5</strong><br>
Used during evolution
</td>

<td align="center">
<a href="media/InvertedPendulum-v5.pdf">
<img src="media/InvertedPendulum-v5-1.png" alt="InvertedPendulum-v5 results" width="100%">
</a>
<br>
<strong>InvertedPendulum-v5</strong><br>
Unseen during evolution
</td>
</tr>

<tr>
<td align="center">
<a href="media/Reacher-v5.pdf">
<img src="media/Reacher-v5-1.png" alt="Reacher-v5 results" width="100%">
</a>
<br>
<strong>Reacher-v5</strong><br>
Unseen during evolution
</td>

<td align="center">
<a href="media/Swimmer-v5.pdf">
<img src="media/Swimmer-v5-1.png" alt="Swimmer-v5 results" width="100%">
</a>
<br>
<strong>Swimmer-v5</strong><br>
Unseen during evolution
</td>
</tr>

<tr>
<td align="center">
<a href="media/Walker2d-v5.pdf">
<img src="media/Walker2d-v5-1.png" alt="Walker2d-v5 results" width="100%">
</a>
<br>
<strong>Walker2d-v5</strong><br>
Unseen during evolution
</td>

<td align="center">
<a href="media/Pusher-v5.pdf">
<img src="media/Pusher-v5-1.png" alt="Pusher-v5 results" width="100%">
</a>
<br>
<strong>Pusher-v5</strong><br>
Unseen during evolution
</td>
</tr>
</table>

Click any figure to open its full-resolution PDF.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sygkounas/evolutionary_discovery.git
cd evolutionary_discovery
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The package versions used for the experiments are specified in `requirements.txt`.

---

## Repository media

```text
media/
├── framework_overview_rl-1.png
├── framework_overview_rl.pdf
├── max_fitness_gpt_vs_claude-1.png
├── max_fitness_gpt_vs_claude.pdf
├── ablation_alpha_0_vs_1.pdf
├── CartPole-v1-1.png
├── CartPole-v1.pdf
├── LunarLander-v3-1.png
├── LunarLander-v3.pdf
├── MountainCar-v0-1.png
├── MountainCar-v0.pdf
├── Acrobot-v1-1.png
├── Acrobot-v1.pdf
├── HalfCheetah-v5-1.png
├── HalfCheetah-v5.pdf
├── InvertedPendulum-v5-1.png
├── InvertedPendulum-v5.pdf
├── Reacher-v5-1.png
├── Reacher-v5.pdf
├── Swimmer-v5-1.png
├── Swimmer-v5.pdf
├── Walker2d-v5-1.png
├── Walker2d-v5.pdf
├── Pusher-v5-1.png
├── Pusher-v5.pdf
└── Designing_and_Evolving_New_Reinforcement_Learning_Algorithms_using_Large_Language_models (4).pdf
```

---

## Paper

- [Conference paper PDF](media/Designing_and_Evolving_New_Reinforcement_Learning_Algorithms_using_Large_Language_models%20%284%29.pdf)
- [ACM Digital Library](https://doi.org/10.1145/3795095.3805180)

---

## Citation

```bibtex
@inproceedings{sygkounas2026evolutionary,
  title     = {Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models},
  author    = {Sygkounas, Alkis and Loutfi, Amy and Persson, Andreas},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3795095.3805180}
}
```

This work builds on REvolve:

```bibtex
@inproceedings{hazra2025revolve,
  title     = {{RE}volve: Reward Evolution with Large Language Models using Human Feedback},
  author    = {Hazra, Rishi and Sygkounas, Alkis and Persson, Andreas and Loutfi, Amy and Zuidberg Dos Martires, Pedro},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=cJPUpL8mOw}
}
```

---

## Acknowledgements

This work is supported by the Knut and Alice Wallenberg Foundation through the Wallenberg AI, Autonomous Systems and Software Program and the Wallenberg Scholars Grant.

Computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden through the LUMI supercomputer.
