import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_max_profit(plants, jobs, compatibility):
    """
    Maximize total profit using the Hungarian algorithm.

    Incompatible assignments are forbidden.
    Dummy columns allow plants to remain unassigned.
    """

    n_plants = len(plants)
    n_jobs = len(jobs)

    # Add dummy jobs so plants can remain unassigned
    n_columns = n_jobs + n_plants

    # Use a profit matrix with a large negative value
    # for incompatible plant-job pairs.
    forbidden = -1e9

    profit_matrix = np.full(
        (n_plants, n_columns),
        forbidden,
        dtype=float
    )

    # Dummy jobs have zero profit
    profit_matrix[:, n_jobs:] = 0

    plant_index = {plant: i for i, plant in enumerate(plants)}
    job_index = {job: j for j, job in enumerate(jobs)}

    # Insert profits for compatible assignments
    for _, row in compatibility.iterrows():
        i = plant_index[row["plant_id"]]
        j = job_index[row["job_id"]]

        profit_matrix[i, j] = float(row["profit"])

    # Maximize total profit
    row_indices, col_indices = linear_sum_assignment(
        profit_matrix,
        maximize=True
    )

    matching = {}

    for i, j in zip(row_indices, col_indices):
        # Ignore dummy jobs
        if j >= n_jobs:
            continue

        # Ignore incompatible assignments
        if profit_matrix[i, j] == forbidden:
            continue

        matching[plants[i]] = jobs[j]

    return matching