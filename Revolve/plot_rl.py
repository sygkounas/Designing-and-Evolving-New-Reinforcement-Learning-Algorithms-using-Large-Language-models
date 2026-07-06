import os, glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

ROOT = "/home/alkis/Downloads/new_rl_stuff/labbet_runs/Revolve/Revolve/database/revolve_auto/10/island_0/rl_logs/9_4_mountain_car/9_4/cfg_8/training_final"
ENV = "MountainCar-v0"

# find all seed event files
pattern = os.path.join(ROOT, ENV, "seed_*", "events.out.tfevents.*")
event_files = sorted(glob.glob(pattern))
print("Found:", len(event_files), "event files")
for f in event_files:
    print(" -", f)

# tags to try (some may not exist)
CANDIDATE_TAGS = [
    "train/reward",
    "eval/mean_reward",
    "train/loss",
    "train/grad_norm",
    "train/param_norm",
]

def load_scalars(event_file):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return ea, set(tags)

def plot_tag(tag):
    plt.figure()
    any_found = False
    for f in event_files:
        ea, tags = load_scalars(f)
        if tag not in tags:
            continue
        any_found = True
        seed = os.path.basename(os.path.dirname(f))
        xs, ys = [], []
        for e in ea.Scalars(tag):
            xs.append(e.step)
            ys.append(e.value)
        plt.plot(xs, ys, label=seed)
    if not any_found:
        plt.close()
        print(f"[skip] tag not found: {tag}")
        return
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel(tag)
    plt.legend()
    plt.tight_layout()
    out = f"cfg8_{tag.replace('/','_')}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved", out)

# print available tags from first file
if event_files:
    ea0, tags0 = load_scalars(event_files[0])
    print("\nAvailable scalar tags (from first seed):")
    for t in sorted(tags0):
        print(" ", t)

for tag in CANDIDATE_TAGS:
    plot_tag(tag)
