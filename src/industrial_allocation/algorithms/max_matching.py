def maximum_bipartite_matching(plants, compatibility):
    """
    Find a maximum-cardinality matching using
    DFS-based augmenting paths (Kuhn's algorithm).
    """

    # Build adjacency list: each plant's compatible jobs
    adjacency = {plant: [] for plant in plants}

    for _, row in compatibility.iterrows():
        adjacency[row["plant_id"]].append(row["job_id"])

    # Store the plant assigned to each job
    job_match = {}

    def dfs(plant, visited_jobs):
        for job in adjacency[plant]:
            if job in visited_jobs:
                continue

            visited_jobs.add(job)

            # Assign if free, or recursively move its current plant
            if (
                job not in job_match
                or dfs(job_match[job], visited_jobs)
            ):
                job_match[job] = plant
                return True

        return False

    # Try to match every plant
    for plant in plants:
        visited_jobs = set()
        dfs(plant, visited_jobs)

    # Convert job -> plant to plant -> job
    return {
        plant: job
        for job, plant in job_match.items()
    }