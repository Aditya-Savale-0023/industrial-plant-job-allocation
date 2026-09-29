import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from industrial_allocation.utils.data_loader import load_data
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Allocation",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Industrial Plant–Job Allocation")
st.caption("MFCS Project | Matching algorithms and optimization")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    plants, jobs, compatibility = load_data()

    required_plant_columns = {
        "plant_id",
        "plant_name",
        "capacity",
    }
    required_job_columns = {
        "job_id",
    }
    required_compatibility_columns = {
        "plant_id",
        "job_id",
        "profit",
    }

    if not required_plant_columns.issubset(plants.columns):
        raise ValueError(
            "The plants data is missing required columns: "
            f"{required_plant_columns - set(plants.columns)}"
        )

    if not required_job_columns.issubset(jobs.columns):
        raise ValueError(
            "The jobs data is missing the job_id column."
        )

    if not required_compatibility_columns.issubset(
        compatibility.columns
    ):
        raise ValueError(
            "The compatibility data is missing required columns: "
            f"{required_compatibility_columns - set(compatibility.columns)}"
        )

    if plants["plant_id"].duplicated().any():
        raise ValueError("Duplicate plant IDs found.")

    if jobs["job_id"].duplicated().any():
        raise ValueError("Duplicate job IDs found.")

    if compatibility.duplicated(
        subset=["plant_id", "job_id"]
    ).any():
        raise ValueError(
            "Duplicate plant-job compatibility pairs found."
        )

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


plant_ids = plants["plant_id"].tolist()
job_ids = jobs["job_id"].tolist()


# --------------------------------------------------
# NORMALIZE ALGORITHM OUTPUTS
# --------------------------------------------------

def normalize_matching(raw_matching):
    """
    Convert algorithm outputs into a common format:

    {
        plant_id: [job_id, job_id, ...]
    }

    One-to-one algorithms return one job per plant.
    Min-cost flow returns a list of jobs per plant.
    """

    if not isinstance(raw_matching, dict):
        raise TypeError(
            "The selected algorithm must return a dictionary. "
            f"Received: {type(raw_matching).__name__}"
        )

    matching = {}

    for plant, assigned in raw_matching.items():

        if assigned is None:
            matching[plant] = []

        elif isinstance(assigned, (list, tuple, set)):
            matching[plant] = list(assigned)

        else:
            matching[plant] = [assigned]

    return matching


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Configuration")

algorithm = st.sidebar.selectbox(
    "Select allocation algorithm",
    [
        "Gale-Shapley",
        "Maximum Bipartite Matching",
        "Hungarian Algorithm",
        "Capacitated Min-Cost Flow",
    ],
)

st.sidebar.caption(
    "Choose an algorithm to recalculate the allocation."
)


# --------------------------------------------------
# RUN SELECTED ALGORITHM
# --------------------------------------------------

try:

    if algorithm == "Gale-Shapley":

        raw_matching = gale_shapley(
            plant_ids,
            job_ids,
            compatibility,
        )

    elif algorithm == "Maximum Bipartite Matching":

        raw_matching = maximum_bipartite_matching(
            plant_ids,
            compatibility,
        )

    elif algorithm == "Hungarian Algorithm":

        raw_matching = hungarian_max_profit(
            plant_ids,
            job_ids,
            compatibility,
        )

    else:

        raw_matching = capacitated_max_profit(
            plants,
            job_ids,
            compatibility,
        )

    matching = normalize_matching(raw_matching)

    # Check that every returned plant and job exists.
    invalid_plants = set(matching) - set(plant_ids)

    if invalid_plants:
        raise ValueError(
            f"Algorithm returned unknown plants: {invalid_plants}"
        )

    assigned_jobs = [
        job
        for assigned in matching.values()
        for job in assigned
    ]

    invalid_jobs = set(assigned_jobs) - set(job_ids)

    if invalid_jobs:
        raise ValueError(
            f"Algorithm returned unknown jobs: {invalid_jobs}"
        )

    # Each job must be assigned at most once.
    if len(assigned_jobs) != len(set(assigned_jobs)):
        raise ValueError(
            "An algorithm assigned the same job to multiple plants."
        )

    # Each plant must not exceed its capacity.
    for _, plant_row in plants.iterrows():
        plant_id = plant_row["plant_id"]
        capacity = int(plant_row["capacity"])

        if len(matching.get(plant_id, [])) > capacity:
            raise ValueError(
                f"{plant_id} exceeds its capacity of {capacity}."
            )

except Exception as error:
    st.error(f"Allocation failed: {error}")
    st.stop()


# --------------------------------------------------
# CALCULATE ALLOCATION METRICS
# --------------------------------------------------

job_to_plant = {
    job: plant
    for plant, assigned in matching.items()
    for job in assigned
}

profit_lookup = {
    (row["plant_id"], row["job_id"]): float(row["profit"])
    for _, row in compatibility.iterrows()
}

total_profit = sum(
    profit_lookup[(plant, job)]
    for plant, assigned in matching.items()
    for job in assigned
)

matched_count = len(job_to_plant)
unmatched_count = len(job_ids) - matched_count


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Profit",
    f"{total_profit:,.2f}",
)

col2.metric(
    "Jobs Assigned",
    f"{matched_count}/{len(job_ids)}",
)

col3.metric(
    "Unassigned Jobs",
    unmatched_count,
)

col4.metric(
    "Algorithm",
    algorithm,
)

st.divider()


# --------------------------------------------------
# PLANT CAPACITY UTILIZATION
# --------------------------------------------------

st.subheader("Plant Capacity Utilization")

capacity_rows = []

for _, plant in plants.iterrows():

    plant_id = plant["plant_id"]
    assigned = matching.get(plant_id, [])
    capacity = int(plant["capacity"])

    plant_profit = sum(
        profit_lookup[(plant_id, job)]
        for job in assigned
    )

    available_capacity = capacity - len(assigned)

    utilization = (
        len(assigned) / capacity * 100
        if capacity > 0
        else 0
    )

    capacity_rows.append(
        {
            "Plant": plant_id,
            "Name": plant["plant_name"],
            "Capacity": capacity,
            "Assigned Jobs": len(assigned),
            "Available Capacity": available_capacity,
            "Utilization (%)": utilization,
            "Profit": plant_profit,
        }
    )

capacity_df = pd.DataFrame(capacity_rows)

st.dataframe(
    capacity_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Utilization (%)": st.column_config.ProgressColumn(
            "Utilization (%)",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "Profit": st.column_config.NumberColumn(
            "Profit",
            format="%.2f",
        ),
    },
)


# --------------------------------------------------
# PLANT-JOB ASSIGNMENT NETWORK
# --------------------------------------------------

st.subheader("Plant–Job Assignment Network")

st.caption(
    "Gray lines show compatible plant-job pairs. "
    "Green lines show selected assignments."
)

G = nx.Graph()

# Add plant nodes.
for _, plant in plants.iterrows():

    plant_id = plant["plant_id"]

    G.add_node(
        plant_id,
        bipartite="plant",
        label=(
            f"{plant_id}\n"
            f"Capacity: {int(plant['capacity'])}"
        ),
    )

# Add job nodes.
for _, job in jobs.iterrows():

    job_id = job["job_id"]

    G.add_node(
        job_id,
        bipartite="job",
        label=str(job_id),
    )

# Add compatible assignment edges.
for _, edge in compatibility.iterrows():

    G.add_edge(
        edge["plant_id"],
        edge["job_id"],
        profit=float(edge["profit"]),
    )

# Custom two-column layout.
pos = {}

number_of_plants = len(plant_ids)
number_of_jobs = len(job_ids)

for i, plant_id in enumerate(plant_ids):

    y = (
        1 - i / max(number_of_plants - 1, 1)
        if number_of_plants > 1
        else 0.5
    )

    pos[plant_id] = (0, y)

for i, job_id in enumerate(job_ids):

    y = (
        1 - i / max(number_of_jobs - 1, 1)
        if number_of_jobs > 1
        else 0.5
    )

    pos[job_id] = (1, y)

# Adjust plot height for larger datasets.
fig_height = max(
    8,
    min(24, number_of_jobs * 0.22),
)

fig, ax = plt.subplots(
    figsize=(14, fig_height),
)

# Draw plant nodes.
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=plant_ids,
    node_color="lightblue",
    node_size=1900,
    ax=ax,
)

# Draw job nodes.
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=job_ids,
    node_color="lightyellow",
    node_size=450,
    ax=ax,
)

# Draw all compatible edges.
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=list(G.edges()),
    edge_color="lightgray",
    width=0.5,
    alpha=0.4,
    ax=ax,
)

# Draw selected edges.
selected_edges = [
    (plant_id, job_id)
    for plant_id, assigned in matching.items()
    for job_id in assigned
]

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=selected_edges,
    edge_color="green",
    width=2,
    ax=ax,
)

# Draw plant labels.
plant_labels = {
    plant_id: G.nodes[plant_id]["label"]
    for plant_id in plant_ids
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=plant_labels,
    font_size=9,
    ax=ax,
)

# Draw job labels.
job_labels = {
    job_id: str(job_id)
    for job_id in job_ids
}

nx.draw_networkx_labels(
    G,
    pos,
    labels=job_labels,
    font_size=6,
    ax=ax,
)

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.04, 1.04)
ax.set_axis_off()

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)


# --------------------------------------------------
# ALLOCATION DETAILS
# --------------------------------------------------

st.subheader("Allocation Details")

assignment_rows = []

for job_id in job_ids:

    plant_id = job_to_plant.get(job_id)

    if plant_id is not None:

        profit = profit_lookup[
            (plant_id, job_id)
        ]

        status = "Assigned"

    else:

        profit = 0
        status = "Unassigned"

    assignment_rows.append(
        {
            "Job": job_id,
            "Assigned Plant": (
                plant_id
                if plant_id is not None
                else "Unassigned"
            ),
            "Profit": profit,
            "Status": status,
        }
    )

assignment_df = pd.DataFrame(assignment_rows)

st.dataframe(
    assignment_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Profit": st.column_config.NumberColumn(
            "Profit",
            format="%.2f",
        ),
    },
)

# Show unassigned jobs explicitly.
unassigned_jobs = [
    job_id
    for job_id in job_ids
    if job_id not in job_to_plant
]

if unassigned_jobs:

    with st.expander(
        f"View {len(unassigned_jobs)} unassigned jobs"
    ):
        st.write(unassigned_jobs)

else:

    st.success("All jobs have been assigned.")


# --------------------------------------------------
# DOWNLOAD REPORTS
# --------------------------------------------------

st.subheader("Download Allocation Report")

col1, col2 = st.columns(2)

with col1:

    csv_data = assignment_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download assignment CSV",
        data=csv_data,
        file_name="industrial_allocation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:

    capacity_csv = capacity_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download capacity utilization CSV",
        data=capacity_csv,
        file_name="plant_capacity_report.csv",
        mime="text/csv",
        use_container_width=True,
    )