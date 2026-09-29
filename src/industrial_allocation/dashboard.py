
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
# NORMALIZE THE OUTPUT OF EACH ALGORITHM
# --------------------------------------------------

def normalize_matching(raw_matching, valid_jobs):
    """
    Convert algorithm outputs into one common format:
        {plant_id: [job_id, job_id, ...]}

    Handles:
        {plant_id: job_id}
        {plant_id: [job_id, ...]}

    Numeric job IDs are treated as single jobs,
    not as iterable collections.
    """

    if raw_matching is None:
        return {}

    if not isinstance(raw_matching, dict):
        raise TypeError(
            "The allocation algorithm must return a dictionary. "
            f"Received: {type(raw_matching).__name__}"
        )

    matching = {}
    assigned_jobs = set()

    for plant, assigned in raw_matching.items():

        if assigned is None:
            continue

        # A single job ID (including an integer) becomes a list.
        if isinstance(assigned, (list, tuple, set)):
            jobs_for_plant = list(assigned)
        else:
            jobs_for_plant = [assigned]

        matching[plant] = []

        for job in jobs_for_plant:

            # Skip common unassigned markers.
            if job is None or job == -1:
                continue

            if isinstance(job, str) and job.strip() == "":
                continue

            if job not in valid_jobs:
                raise ValueError(
                    f"Algorithm assigned unknown job {job!r} "
                    f"to plant {plant!r}."
                )

            if job in assigned_jobs:
                raise ValueError(
                    f"Job {job!r} was assigned to more than one plant."
                )

            matching[plant].append(job)
            assigned_jobs.add(job)

        # Remove plants with no assigned jobs.
        if not matching[plant]:
            del matching[plant]

    return matching


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    plants, jobs, compatibility = load_data()

    required_plant_cols = {
        "plant_id", "plant_name", "capacity"
    }
    required_job_cols = {"job_id"}
    required_compat_cols = {
        "plant_id", "job_id", "profit"
    }

    if not required_plant_cols.issubset(plants.columns):
        raise ValueError(
            f"Plants data must contain: {required_plant_cols}"
        )

    if not required_job_cols.issubset(jobs.columns):
        raise ValueError(
            f"Jobs data must contain: {required_job_cols}"
        )

    if not required_compat_cols.issubset(compatibility.columns):
        raise ValueError(
            f"Compatibility data must contain: {required_compat_cols}"
        )

    plant_ids = plants["plant_id"].tolist()
    job_ids = jobs["job_id"].tolist()

    if not plant_ids or not job_ids:
        st.error("No plants or jobs were found in the loaded data.")
        st.stop()

    valid_jobs = set(job_ids)

    # Check for duplicate IDs.
    if len(plant_ids) != len(set(plant_ids)):
        raise ValueError("Duplicate plant IDs found.")

    if len(job_ids) != len(valid_jobs):
        raise ValueError("Duplicate job IDs found.")

    # Each plant-job pair should occur only once.
    if compatibility.duplicated(
        subset=["plant_id", "job_id"]
    ).any():
        raise ValueError(
            "Duplicate plant-job compatibility records found."
        )

    # Use numeric profit values.
    compatibility = compatibility.copy()
    compatibility["profit"] = pd.to_numeric(
        compatibility["profit"], errors="raise"
    )

except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.exception(exc)
    st.stop()


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
# RUN THE SELECTED ALGORITHM
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

    matching = normalize_matching(
        raw_matching,
        valid_jobs,
    )

except Exception as exc:
    st.error(f"Allocation failed: {exc}")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# VALIDATE ASSIGNMENTS AND CALCULATE METRICS
# --------------------------------------------------

profit_lookup = {
    (row["plant_id"], row["job_id"]): float(row["profit"])
    for _, row in compatibility.iterrows()
}

compatible_pairs = set(profit_lookup.keys())

for plant, assigned_jobs in matching.items():

    if plant not in plant_ids:
        st.error(f"Unknown plant in allocation: {plant!r}")
        st.stop()

    capacity = int(
        plants.loc[
            plants["plant_id"] == plant,
            "capacity",
        ].iloc[0]
    )

    if len(assigned_jobs) > capacity:
        st.error(
            f"{plant} exceeds its capacity of {capacity}."
        )
        st.stop()

    for job in assigned_jobs:
        if (plant, job) not in compatible_pairs:
            st.error(
                f"Incompatible assignment detected: "
                f"{plant} → {job}"
            )
            st.stop()

job_to_plant = {
    job: plant
    for plant, assigned_jobs in matching.items()
    for job in assigned_jobs
}

total_profit = sum(
    profit_lookup[(plant, job)]
    for plant, assigned_jobs in matching.items()
    for job in assigned_jobs
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

    utilization = (
        len(assigned) / capacity * 100
        if capacity > 0
        else 0
    )

    capacity_rows.append({
        "Plant": plant_id,
        "Name": plant["plant_name"],
        "Capacity": capacity,
        "Assigned Jobs": len(assigned),
        "Available Capacity": capacity - len(assigned),
        "Utilization (%)": utilization,
        "Profit": plant_profit,
    })

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
        )
    },
)


# --------------------------------------------------
# ASSIGNMENT NETWORK
# --------------------------------------------------

st.subheader("Plant–Job Assignment Network")

st.caption(
    "Gray lines show compatible assignments. "
    "Green lines show selected assignments."
)

G = nx.Graph()

# Add plant nodes.
for _, plant in plants.iterrows():

    plant_id = plant["plant_id"]

    G.add_node(
        plant_id,
        bipartite=0,
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
        bipartite=1,
        label=str(job_id),
    )

# Add compatible plant-job edges.
for _, edge in compatibility.iterrows():

    G.add_edge(
        edge["plant_id"],
        edge["job_id"],
        profit=float(edge["profit"]),
    )

# Two-column layout.
pos = {}

for i, plant_id in enumerate(plant_ids):

    y = (
        1 - i / (len(plant_ids) - 1)
        if len(plant_ids) > 1
        else 0.5
    )

    pos[plant_id] = (0, y)

for i, job_id in enumerate(job_ids):

    y = (
        1 - i / (len(job_ids) - 1)
        if len(job_ids) > 1
        else 0.5
    )

    pos[job_id] = (1, y)

# Limit figure height for large datasets.
fig_height = max(
    8,
    min(30, len(job_ids) * 0.20),
)

fig, ax = plt.subplots(
    figsize=(14, fig_height)
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

# Draw all compatible edges in gray.
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=list(G.edges()),
    edge_color="lightgray",
    width=0.5,
    alpha=0.4,
    ax=ax,
)

# Draw selected edges in green.
selected_edges = [
    (plant, job)
    for plant, assigned_jobs in matching.items()
    for job in assigned_jobs
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

# Draw job IDs.
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

for job in job_ids:

    plant = job_to_plant.get(job)

    assignment_rows.append({
        "Job": job,
        "Assigned Plant": (
            plant if plant is not None else "Unassigned"
        ),
        "Profit": (
            profit_lookup[(plant, job)]
            if plant is not None
            else 0
        ),
        "Status": (
            "Assigned"
            if plant is not None
            else "Unassigned"
        ),
    })

assignment_df = pd.DataFrame(assignment_rows)

st.dataframe(
    assignment_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# DOWNLOAD REPORTS
# --------------------------------------------------

st.subheader("Download Allocation Reports")

# 1. Assignment report.
assignment_csv = assignment_df.to_csv(
    index=False
).encode("utf-8")

# 2. Plant capacity report.
capacity_csv = capacity_df.to_csv(
    index=False
).encode("utf-8")

# 3. Overall allocation summary.
summary_df = pd.DataFrame([
    {
        "Algorithm": algorithm,
        "Total Profit": total_profit,
        "Total Jobs": len(job_ids),
        "Jobs Assigned": matched_count,
        "Unassigned Jobs": unmatched_count,
        "Total Plants": len(plant_ids),
    }
])

summary_csv = summary_df.to_csv(
    index=False
).encode("utf-8")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="Download Assignment CSV",
        data=assignment_csv,
        file_name="assignment_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col2:
    st.download_button(
        label="Download Capacity CSV",
        data=capacity_csv,
        file_name="plant_capacity_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col3:
    st.download_button(
        label="Download Allocation Summary",
        data=summary_csv,
        file_name="industrial_allocation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )