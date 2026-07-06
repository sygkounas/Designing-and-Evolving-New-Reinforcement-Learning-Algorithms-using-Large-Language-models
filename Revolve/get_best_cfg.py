 python - << 'EOF'
> import json
> from pathlib import Path
> 
> base = Path(".")
> best_cfg = None
> best_val = -1e9
> 
> print("CFG      AVG_MEAN_REWARD")
> 
> for cfg in sorted(base.glob("cfg_*")):
>     f = cfg / "training_final" / "final_training.json"
>     if not f.exists():
>         continue
> 
>     with open(f) as fh:
>         data = json.load(fh)
> 
>     env_stats = data.get("env_seed_stats", {})
>     if not env_stats:
>         continue
> 
>     # single env (CartPole-v1)
>     seeds = next(iter(env_stats.values()))
>     means = [v["mean_reward"] for v in seeds.values()]
>     if not means:
>         continue
> 
>     avg = sum(means) / len(means)
>     print(f"{cfg.name:8s} {avg:.3f}")
> 
>     if avg > best_val:
>         best_val = avg
>         best_cfg = cfg.name
> 
> print("\nBEST CFG:", best_cfg, "AVG_MEAN_REWARD =", round(best_val, 3))
> EOF
