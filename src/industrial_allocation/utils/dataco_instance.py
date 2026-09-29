
from pathlib import Path
import math

import pandas as pd

from industrial_allocation.utils.dataco_loader import (
    load_dataco_orders,
)


def create_allocation_instance(
    n_jobs=100,
    n_hubs=5,
    capacity_fraction=0.80,
    transport_penalty=15.0,
    random_state=42,
):
    """
    Build a small allocation experiment from historical DataCo data.

    Hubs, capacities, and transport penalties are modeled
    assumptions. Historical regions/categories/profit ratios
    inform the experiment, not actual hub operations.
    """

    if n_jobs < 1 or n_hubs < 1:
        raise ValueError("n_jobs and n_hubs must be positive.")

    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1].")

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned historical order items.
    orders = load_dataco_orders()

    # Select the five most frequent historical regions.
    top_regions = (
        orders["region"]
        .value_counts()
        .head(n_hubs)
        .index
        .tolist()
    )

    # Restrict the experimental sample to those regions.
    eligible_orders = orders[
        orders["region"].isin(top_regions)
    ].copy()

    if eligible_orders.empty:
        raise ValueError("No orders available in selected regions.")

    # Sample deterministically so comparisons are reproducible.
    jobs = eligible_orders.sample(
        n=min(n_jobs, len(eligible_orders)),
        random_state=random_state,
    ).copy()

    jobs = jobs.reset_index(drop=True)
    jobs["job_id"] = jobs["job_id"].astype(str)

    # Each hypothetical hub serves one historical region.
    hubs = pd.DataFrame({
        "hub_id": [f"H{i + 1:02d}" for i in range(len(top_regions))],
        "region": top_regions,
    })

    # Capacity is measured in number of order items,
    # not units, weight, or processing hours.
    total_capacity = max(
        1, math.floor(len(jobs) * capacity_fraction)
    )

    base_capacity, remainder = divmod(
        total_capacity, len(hubs)
    )

    hubs["capacity"] = [
        base_capacity + (1 if i < remainder else 0)
        for i in range(len(hubs))
    ]

    # Build observed category-region combinations using
    # the complete historical dataset.
    observed_pairs = set(
        zip(orders["region"], orders["category"])
    )

    # Candidate assignments are based on historical evidence
    # that a category occurred in a region.
    compatibility_rows = []

    for job in jobs.itertuples(index=False):
        for hub in hubs.itertuples(index=False):

            pair = (hub.region, job.category)

            if pair not in observed_pairs:
                continue

            same_region = job.region == hub.region

            # Assumed penalty for serving a job from a
            # different region. This is not actual freight cost.
            penalty = (
                0.0 if same_region else transport_penalty
            )

            compatibility_rows.append({
                "hub_id": hub.hub_id,
                "job_id": job.job_id,
                "historical_profit_proxy": job.estimated_profit,
                "assumed_transport_penalty": penalty,
                "assignment_score": (
                    job.estimated_profit - penalty
                ),
                "same_region": same_region,
            })

    compatibility = pd.DataFrame(
        compatibility_rows,
        columns=[
            "hub_id",
            "job_id",
            "historical_profit_proxy",
            "assumed_transport_penalty",
            "assignment_score",
            "same_region",
        ],
    )

    if compatibility.empty:
        raise ValueError("No compatible hub-job pairs were generated.")

    
    # Normalize DataCo data to the existing algorithm schema.
    # "Plant" here represents a hypothetical fulfillment hub.

    hubs["plant_id"] = hubs["hub_id"]
    hubs["plant_name"] = hubs["region"]

    jobs["job_name"] = (
        jobs["category"].astype(str)
        + " | "
        + jobs["region"].astype(str)
    )

    compatibility["plant_id"] = compatibility["hub_id"]
    compatibility["profit"] = compatibility["assignment_score"]

    # Save all three input tables.
    jobs.to_csv(output_dir / "jobs.csv", index=False)
    hubs.to_csv(output_dir / "hubs.csv", index=False)
    compatibility.to_csv(
        output_dir / "compatibility.csv",
        index=False,
    )

    print("\nDATACO ALLOCATION INSTANCE")
    print("-" * 40)
    print(f"Historical order items sampled: {len(jobs)}")
    print(f"Hypothetical hubs: {len(hubs)}")
    print(f"Candidate assignments: {len(compatibility)}")
    print(f"Modeled total capacity: {hubs['capacity'].sum()}")
    print(f"Capacity per hub:\n{hubs[['hub_id', 'region', 'capacity']].to_string(index=False)}")
    print("\nFiles saved to:", output_dir)

    return jobs, hubs, compatibility


if __name__ == "__main__":
    create_allocation_instance()