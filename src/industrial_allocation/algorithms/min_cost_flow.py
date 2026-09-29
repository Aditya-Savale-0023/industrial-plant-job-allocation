import networkx as nx


def capacitated_max_profit(plants, jobs, compatibility):
    """
    Maximize total profit using min-cost flow.

    Constraints:
    - Each job is assigned to at most one plant.
    - Each plant can take up to its capacity.
    - Incompatible assignments are forbidden.
    - Jobs may remain unassigned.
    """

    G = nx.DiGraph()

    source = "SOURCE"
    sink = "SINK"

    # Every job sends exactly one unit of flow
    G.add_node(source, demand=-len(jobs))
    G.add_node(sink, demand=len(jobs))

    # Source -> jobs
    for job in jobs:
        G.add_edge(
            source,
            job,
            capacity=1,
            weight=0
        )

    # Plants -> sink, limited by capacity
    for _, plant in plants.iterrows():
        G.add_edge(
            plant["plant_id"],
            sink,
            capacity=int(plant["capacity"]),
            weight=0
        )

    # Compatible job -> plant edges
    # Negative cost = profit maximization
    for _, row in compatibility.iterrows():
        job = row["job_id"]
        plant = row["plant_id"]

        # Scale profits to integer cents for the flow solver
        profit_cents = int(round(float(row["profit"]) * 100))

        G.add_edge(
            job,
            plant,
            capacity=1,
            weight=-profit_cents
        )

    # Each job can also be left unassigned at zero cost
    for job in jobs:
        G.add_edge(
            job,
            sink,
            capacity=1,
            weight=0
        )

    # Solve the minimum-cost flow
    flow = nx.min_cost_flow(G)

    # Extract job -> plant assignments
    matching = {}

    for job in jobs:
        for plant in plants["plant_id"]:
            if flow[job].get(plant, 0) == 1:
                matching[plant] = matching.get(plant, []) + [job]

    return matching