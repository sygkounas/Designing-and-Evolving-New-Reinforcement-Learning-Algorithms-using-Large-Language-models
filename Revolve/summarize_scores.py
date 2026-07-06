import os
from collections import defaultdict
import heapq


def analyze_population(db_path):
    islands = [d for d in os.listdir(db_path) if d.startswith("island_")]
    islands = sorted(islands, key=lambda x: int(x.split("_")[1]))

    global_pop = defaultdict(list)
    global_fitness = defaultdict(list)

    # top-5 as min-heap: (fitness, path)
    top5 = []

    print("\n================ PER-ISLAND STATS ================\n")

    for isl in islands:
        isl_path = os.path.join(db_path, isl)

        gen_pop = defaultdict(int)
        gen_fit = defaultdict(list)

        gen_dir = os.path.join(isl_path, "generated_fns")
        fit_dir = os.path.join(isl_path, "fitness_scores")

        if os.path.exists(gen_dir):
            for f in os.listdir(gen_dir):
                if "_" not in f:
                    continue
                gen = f.split("_")[0]
                gen_pop[gen] += 1
                global_pop[gen].append(f)

        if os.path.exists(fit_dir):
            for f in os.listdir(fit_dir):
                if "_" not in f:
                    continue
                gen = f.split("_")[0]
                fpath = os.path.join(fit_dir, f)
                try:
                    with open(fpath, "r") as ff:
                        fit = float(ff.read().strip())

                        gen_fit[gen].append(fit)
                        global_fitness[gen].append(fit)

                        # --- update top-5 ---
                        if len(top5) < 6:
                            heapq.heappush(top5, (fit, fpath))
                        else:
                            heapq.heappushpop(top5, (fit, fpath))

                except:
                    pass

        print(f"Island {isl}:")
        gens = sorted(set(gen_pop) | set(gen_fit), key=lambda x: int(x))

        for gen in gens:
            fits = gen_fit.get(gen, [])
            avg_fit = sum(fits) / len(fits) if fits else 0
            max_fit = max(fits) if fits else 0
            print(f"  Generation {gen}: pop={gen_pop.get(gen,0)}, avg_fit={avg_fit:.4f}, max_fit={max_fit:.4f}")

        print("")

    print("\n================ GLOBAL (ALL ISLANDS) ================\n")
    for gen in sorted(global_pop, key=lambda x: int(x)):
        fits = global_fitness.get(gen, [])
        avg_fit = sum(fits) / len(fits) if fits else 0
        max_fit = max(fits) if fits else 0
        print(f"Generation {gen}: pop={len(global_pop[gen])}, avg_fit={avg_fit:.4f}, max_fit={max_fit:.4f}")

    print("\n================ TOP-5 INDIVIDUALS ================\n")
    for rank, (fit, path) in enumerate(sorted(top5, reverse=True), 1):
        print(f"{rank}. fitness={fit:.6f}  path={path}")
if __name__ == "__main__":

    # MODIFY THIS PATH ONLY
    DB_PATH = "/home/alkis/Downloads/new_rl_stuff/labbet_runs/Revolve/Revolve/database/revolve_auto/10"

    print(f"Analyzing database at: {DB_PATH}")
    analyze_population(DB_PATH)
