
import time
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_allocation.algorithms.gale_shapley import gale_shapley
from industrial_allocation.algorithms.max_matching import (
    maximum_bipartite_matching,
)
from industrial_allocation.algorithms.hungarian import (
    hungarian_max_profit,
)
from industrial_allocation.algorithms.min_cost_flow import (
    capacitated_max_profit,
)


def generate_dataset(n, seed=42, density=0.6):
    """Generate a reproducible random industrial allocation dataset."""

    rng = np.random.default_rng(seed)

    plants = pd.DataFrame({
        "plant_id": [f"P{i}" for i in range(1, n + 1)],
        "plant_name": [f"Plant {i}" for i in range(1, n + 1)],
        "capacity": rng.integers(2, 5, size=n),
    })

    jobs = pd.DataFrame({
        "job_id": [f"J{i}" for i in range(1, n + 1)],
        "job_name": [f"Job {i}" for i in range(1, n + 1)],
    })

    edges = []

    for plant in plants["plant_id"]:
        for job in jobs["job_id"]:
            if rng.random() < density:
                edges.append({
                    "plant_id": plant,
                    "job_id": job,
                    "profit": int(rng.integers(40, 151)),
                })

    compatibility = pd.DataFrame(
        edges,
        columns=["plant_id", "job_id", "profit"],
    )

    return plants, jobs, compatibility


def evaluate(matching, compatibility):
    """Calculate number of assignments and total profit."""

    total_jobs = 0
    total_profit = 0

    for plant, assigned in matching.items():
        if isinstance(assigned, str):
            assigned = [assigned]

        for job in assigned:
            rows = compatibility[
                (compatibility["plant_id"] == plant)
                & (compatibility["job_id"] == job)
            ]

            if rows.empty:
                raise ValueError(
                    f"Invalid assignment: {plant} -> {job}"
                )

            total_jobs += 1
            total_profit += float(rows.iloc[0]["profit"])

    return total_jobs, total_profit


def run_benchmark():
    sizes = [10, 20, 40, 80]
    trials = 3
    density = 0.6

    algorithms = {
        "Gale-Shapley": lambda p, j, c: gale_shapley(p, j, c),
        "Maximum Matching": lambda p, j, c: (
            maximum_bipartite_matching(p, c)
        ),
        "Hungarian": lambda p, j, c: (
            hungarian_max_profit(p, j, c)
        ),
        "Min-Cost Max-Flow": lambda p, j, c: (
            capacitated_max_profit(
                pd.DataFrame({
                    "plant_id": p,
                    "capacity": capacities
                }),
                j,
                c,
            )
        ),
    }

    results = []

    for n in sizes:
        print(f"\nBenchmarking {n} plants and {n} jobs...")

        # Use one fixed dataset per size, shared by all algorithms.
        plants, jobs, compatibility = generate_dataset(
            n=n,
            seed=42 + n,
            density=density,
        )

        plant_ids = plants["plant_id"].tolist()
        job_ids = jobs["job_id"].tolist()
        capacities = plants["capacity"].tolist()

        for name, algorithm in algorithms.items():
            times = []
            last_matching = None

            for _ in range(trials):
                start = time.perf_counter()

                last_matching = algorithm(
                    plant_ids,
                    job_ids,
                    compatibility,
                )

                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)

            assigned, profit = evaluate(
                last_matching,
                compatibility,
            )

            results.append({
                "Plants": n,
                "Jobs": n,
                "Compatibility Density": density,
                "Algorithm": name,
                "Matched Jobs": assigned,
                "Total Profit": profit,
                "Average Time (ms)": round(
                    sum(times) / len(times), 3
                ),
            })

    result_df = pd.DataFrame(results)

    print("\nBENCHMARK RESULTS")
    print("=" * 90)
    print(result_df.to_string(index=False))

    # Save the experiment results in the project data folder.
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "benchmark_results.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()