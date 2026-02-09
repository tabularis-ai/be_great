import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseMetric(ABC):
    """Abstract Base Class for all GReaT evaluation metrics.

    All metrics compare real (original) tabular data against synthetically generated data.
    Subclasses must implement the `compute` method.

    Attributes:
        name (str): Human-readable name of the metric
        direction (str): Whether higher ("maximize") or lower ("minimize") values are better
    """

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the human-readable name of the metric."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def direction() -> str:
        """Returns 'maximize' if higher is better, 'minimize' if lower is better."""
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """Compute the metric comparing real and synthetic data.

        Args:
            real_data: The original (real) tabular dataset
            synthetic_data: The synthetically generated tabular dataset

        Returns:
            Dictionary with metric name(s) as keys and computed value(s) as values
        """
        raise NotImplementedError

    @staticmethod
    def _detect_column_types(
        df: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> tp.Tuple[tp.List[str], tp.List[str]]:
        """Auto-detect numerical and categorical columns if not provided.

        Args:
            df: DataFrame to inspect
            num_cols: Explicit list of numerical columns (overrides auto-detection)
            cat_cols: Explicit list of categorical columns (overrides auto-detection)

        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        if num_cols is None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if cat_cols is None:
            cat_cols = [c for c in df.columns if c not in num_cols]
        return num_cols, cat_cols

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name()}', direction='{self.direction()}')"
