"""
Data management for Predictif MCP Server
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any
from datetime import datetime


class DataManager:
    """Manages loaded datasets and their metadata"""

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, Any]] = {}

    def load_dataset(
        self, file_path: str, dataset_name: str | None = None
    ) -> dict[str, Any]:
        """Load a dataset and store its metadata"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if dataset_name is None:
            dataset_name = path.stem

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV files are supported. Got: {path.suffix}")

        df = pd.read_csv(file_path)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        metadata: dict[str, Any] = {
            "name": dataset_name,
            "file_path": str(path.absolute()),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "loaded_at": datetime.now().isoformat(),
            "data": df if memory_mb < 100 else None,
        }

        self.datasets[dataset_name] = metadata
        return metadata

    def get_dataset_summary(self, dataset_name: str) -> dict[str, Any]:
        """Get summary information about a dataset"""
        if dataset_name not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

        metadata = self.datasets[dataset_name]
        df = metadata["data"] or pd.read_csv(metadata["file_path"])

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return {
            "name": metadata["name"],
            "shape": metadata["shape"],
            "columns": metadata["columns"],
            "data_types": metadata["dtypes"],
            "missing_values": metadata["missing_values"],
            "total_missing": sum(metadata["missing_values"].values()),
            "memory_usage_mb": round(metadata["memory_usage"] / (1024 * 1024), 2),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "basic_stats": df.describe().to_dict() if numeric_cols else {},
        }

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        """List all loaded datasets"""
        return {
            name: {
                "shape": meta["shape"],
                "file_path": meta["file_path"],
                "loaded_at": meta["loaded_at"],
            }
            for name, meta in self.datasets.items()
        }


# Global instance
data_manager = DataManager()

