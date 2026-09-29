from collections import deque


def gale_shapley(plants, jobs, compatibility):
    """
    Plant-proposing Gale-Shapley stable matching.

    Each plant proposes to jobs in descending profit order.
    Each job keeps its most-preferred compatible plant.
    Every plant and job can be matched at most once.
    """

    # Build plant preference lists
    plant_preferences = {}

    for plant in plants:
        possible_jobs = compatibility[
            compatibility["plant_id"] == plant
        ].sort_values(
            by="profit",
            ascending=False
        )

        plant_preferences[plant] = possible_jobs[
            "job_id"
        ].tolist()

    # Build job preference rankings
    job_preferences = {}

    for job in jobs:
        possible_plants = compatibility[
            compatibility["job_id"] == job
        ].sort_values(
            by="profit",
            ascending=False
        )

        job_preferences[job] = {
            row["plant_id"]: rank
            for rank, (_, row) in enumerate(
                possible_plants.iterrows()
            )
        }

    # A job can hold at most one plant
    matched_jobs = {}

    # Track the next job each plant will propose to
    next_choice = {plant: 0 for plant in plants}

    # Initially, all plants are free
    free_plants = deque(plants)

    while free_plants:
        plant = free_plants.popleft()

        preferences = plant_preferences[plant]

        # No acceptable jobs left for this plant
        if next_choice[plant] >= len(preferences):
            continue

        # Propose to the next preferred job
        job = preferences[next_choice[plant]]
        next_choice[plant] += 1

        if job not in matched_jobs:
            # Job is currently free
            matched_jobs[job] = plant

        else:
            current_plant = matched_jobs[job]

            # Lower rank means higher preference
            new_rank = job_preferences[job][plant]
            current_rank = job_preferences[job][current_plant]

            if new_rank < current_rank:
                # Job prefers the new plant
                matched_jobs[job] = plant
                free_plants.append(current_plant)
            else:
                # Job rejects the new plant
                free_plants.append(plant)

    # Convert job -> plant into plant -> job
    return {
        plant: job
        for job, plant in matched_jobs.items()
    }