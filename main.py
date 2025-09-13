"""
Predictif MCP Server - ML Model Training and Prediction Tools
"""

from mcp.server.fastmcp import FastMCP
from predictif.tools import register_tools

mcp = FastMCP("Predictif MCP Server", port=3000, stateless_http=True, debug=True)

register_tools(mcp)


import pandas as pd


@mcp.tool(
    title="Describe dataset tool",
    description="Provide a text-based EDA summary for a dataset",
)
def describe_dataset() -> str:
    """
    Generate a text-based EDA summary for a dataset.

    Returns:
    - str: A formatted description of the dataset.
    """
    csv_path = "data/iris/train.csv"
    label_column = "Species"
    df = pd.read_csv(csv_path)

    # Basic dataset info
    n_rows, n_cols = df.shape
    desc = []
    desc.append(f"Dataset Summary")
    desc.append(f"- Path: {csv_path}")
    desc.append(f"- Rows: {n_rows}")
    desc.append(f"- Columns: {n_cols}")
    desc.append("")

    # Label column info
    if label_column and label_column in df.columns:
        desc.append(f"Label Column: `{label_column}`")
        desc.append(f"- Type: {df[label_column].dtype}")
        desc.append(f"- Unique values: {df[label_column].nunique()}")
        desc.append("")
    elif label_column:
        desc.append(f"Label column `{label_column}` not found in dataset.")
        desc.append("")

    # Features summary
    desc.append("Features Overview:")
    for col in df.columns:
        if col == label_column:
            continue

        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "Numerical"
        else:
            col_type = "Categorical"

        desc.append(f"- `{col}`")
        desc.append(f"Type: {col_type}")
        desc.append(f"Unique values: {n_unique}")
        if col_type == "Numerical":
            desc.append(
                f"Mean: {df[col].mean():.3f}, Std: {df[col].std():.3f}, Min: {df[col].min()}, Max: {df[col].max()}"
            )
        else:
            most_common = df[col].value_counts().idxmax()
            freq = df[col].value_counts().max()
            desc.append(f"Most frequent: '{most_common}' ({freq} occurrences)")
        desc.append("")

    return "\n".join(desc)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
