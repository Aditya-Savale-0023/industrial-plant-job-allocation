import networkx as nx


def build_graph(plants, jobs, compatibility):
    G = nx.Graph()

    # Add plant vertices
    for _, plant in plants.iterrows():
        G.add_node(
            plant["plant_id"],
            bipartite="plant",
            name=plant["plant_name"],
            capacity=int(plant["capacity"])
        )

    # Add job vertices
    for _, job in jobs.iterrows():
        G.add_node(
            job["job_id"],
            bipartite="job",
            name=job["job_name"]
        )

    # Add weighted edges for compatible assignments
    for _, assignment in compatibility.iterrows():
        G.add_edge(
            assignment["plant_id"],
            assignment["job_id"],
            profit=float(assignment["profit"])
        )

    return G