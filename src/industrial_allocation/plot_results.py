
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Resolve project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "benchmark_results.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load benchmark results
df = pd.read_csv(DATA_FILE)

# Use a consistent order in all plots
algorithm_order = [
    "Gale-Shapley",
    "Maximum Matching",
    "Hungarian",
    "Min-Cost Max-Flow",
]


def plot_metric(metric, ylabel, title, filename):
    plt.figure(figsize=(10, 6))

    for algorithm in algorithm_order:
        subset = df[df["Algorithm"] == algorithm].sort_values("Plants")

        plt.plot(
            subset["Plants"],
            subset[metric],
            marker="o",
            label=algorithm,
        )

    plt.xlabel("Number of plants (and jobs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df["Plants"].unique()))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = REPORT_DIR / filename
    plt.savefig(output_file, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output_file}")


# 1. Execution time
plot_metric(
    metric="Average Time (ms)",
    ylabel="Average execution time (ms)",
    title="Algorithm Runtime vs Problem Size",
    filename="runtime_comparison.png",
)

# 2. Total profit
plot_metric(
    metric="Total Profit",
    ylabel="Total profit",
    title="Allocation Profit vs Problem Size",
    filename="profit_comparison.png",
)

# 3. Number of jobs matched
plot_metric(
    metric="Matched Jobs",
    ylabel="Number of matched jobs",
    title="Assignment Coverage vs Problem Size",
    filename="matched_jobs_comparison.png",
)