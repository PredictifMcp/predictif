"""
Tools for Predictif MCP Server
"""

import json
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from .data_manager import data_manager


def register_tools(mcp: FastMCP):
    """Register all tools with the MCP server"""

    @mcp.tool(
        title="Load Dataset",
        description="Load a CSV dataset for machine learning analysis",
    )
    def load_dataset(
        file_path: str = Field(description="Path to the CSV dataset file"),
        dataset_name: str = Field(description="Name for the dataset", default=""),
    ) -> str:
        try:
            name = dataset_name if dataset_name else None
            metadata = data_manager.load_dataset(file_path, name)

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Dataset '{metadata['name']}' loaded successfully",
                    "shape": metadata["shape"],
                    "columns": len(metadata["columns"]),
                    "memory_usage_mb": round(metadata["memory_usage"] / 1024 / 1024, 2),
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, indent=2)

    @mcp.tool(
        title="Dataset Summary",
        description="Get detailed summary of a loaded dataset including types, dimensions, and statistics",
    )
    def dataset_summary(
        dataset_name: str = Field(description="Name of the dataset to summarize"),
    ) -> str:
        try:
            summary = data_manager.get_dataset_summary(dataset_name)

            return json.dumps(
                {
                    "dataset_name": summary["name"],
                    "dimensions": {
                        "rows": summary["shape"][0],
                        "columns": summary["shape"][1],
                    },
                    "column_types": summary["data_types"],
                    "missing_data": {
                        "total_missing": summary["total_missing"],
                        "by_column": summary["missing_values"],
                    },
                    "memory_usage_mb": summary["memory_usage_mb"],
                    "column_categories": {
                        "numeric": summary["numeric_columns"],
                        "categorical": summary["categorical_columns"],
                    },
                    "basic_statistics": summary["basic_stats"],
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)}, indent=2)

    @mcp.tool(
        title="List Datasets",
        description="List all currently loaded datasets",
    )
    def list_datasets() -> str:
        datasets = data_manager.list_datasets()
        if not datasets:
            return json.dumps(
                {"message": "No datasets currently loaded", "datasets": []}, indent=2
            )

        return json.dumps(
            {"message": f"Found {len(datasets)} loaded datasets", "datasets": datasets},
            indent=2,
        )
