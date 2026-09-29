
from pathlib import Path

import pandas as pd


def load_dataco_orders():
    """Load and clean DataCo historical order-item records."""

    # Project root: industrial-allocation/
    project_root = Path(__file__).resolve().parents[3]
    csv_path = (
        project_root
        / "data"
        / "raw"
        / "DataCoSupplyChainDataset.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"DataCo dataset not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        low_memory=False,
    )

    # Keep only the fields needed for allocation.
    required_columns = [
        "Order Item Id",
        "Order Id",
        "Order Item Quantity",
        "Order Item Total",
        "Order Item Profit Ratio",
        "Category Name",
        "Order Region",
        "Shipping Mode",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}"
        )

    orders = df[required_columns].copy()

    # Clean numeric columns.
    numeric_columns = [
        "Order Item Quantity",
        "Order Item Total",
        "Order Item Profit Ratio",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ]

    for col in numeric_columns:
        orders[col] = pd.to_numeric(
            orders[col], errors="coerce"
        )

    orders = orders.dropna(
        subset=[
            "Order Item Id",
            "Order Item Quantity",
            "Order Item Total",
            "Order Item Profit Ratio",
            "Category Name",
            "Order Region",
        ]
    )

    # One allocation job per historical order item.
    orders["job_id"] = (
        orders["Order Item Id"].astype(str)
    )

    orders["quantity"] = (
        orders["Order Item Quantity"].astype(int)
    )

    orders["revenue"] = (
        orders["Order Item Total"].astype(float)
    )

    # This is an estimated profit proxy, not a verified
    # plant-specific profit or an actual future outcome.
    orders["estimated_profit"] = (
        orders["revenue"]
        * orders["Order Item Profit Ratio"]
    )

    orders["category"] = orders["Category Name"]
    orders["region"] = orders["Order Region"]
    orders["shipping_mode"] = orders["Shipping Mode"]

    orders["actual_shipping_days"] = (
        orders["Days for shipping (real)"]
    )

    orders["scheduled_shipping_days"] = (
        orders["Days for shipment (scheduled)"]
    )

    # Return a compact, non-PII allocation dataset.
    return orders[
        [
            "job_id",
            "Order Id",
            "quantity",
            "revenue",
            "estimated_profit",
            "category",
            "region",
            "shipping_mode",
            "actual_shipping_days",
            "scheduled_shipping_days",
        ]
    ].reset_index(drop=True)