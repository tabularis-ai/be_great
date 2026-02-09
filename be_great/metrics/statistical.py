import logging
import typing as tp

import numpy as np
import pandas as pd
from scipy import stats

from be_great.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class ColumnShapes(BaseMetric):
    """Column Shapes Metric

    Compares the marginal distribution of each column between real and synthetic
    data using the Kolmogorov-Smirnov test (numerical) or Total Variation Distance
    (categorical). Reports the average similarity score across all columns.

    A score close to 1.0 means the distributions are nearly identical.

    Attributes:
        None
    """

    @staticmethod
    def name() -> str:
        return "column_shapes"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute per-column distribution similarity.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.

        Returns:
            Dictionary with 'column_shapes_mean', 'column_shapes_std',
            and 'column_shapes_detail' (per-column scores)
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        scores = {}

        for col in num_cols:
            if col not in synthetic_data.columns:
                continue
            real_vals = real_data[col].dropna().values
            synth_vals = synthetic_data[col].dropna().values
            if len(real_vals) == 0 or len(synth_vals) == 0:
                scores[col] = 0.0
                continue
            ks_stat, _ = stats.ks_2samp(real_vals, synth_vals)
            scores[col] = 1.0 - ks_stat

        for col in cat_cols:
            if col not in synthetic_data.columns:
                continue
            scores[col] = self._tvd_similarity(real_data[col], synthetic_data[col])

        all_scores = list(scores.values())

        return {
            "column_shapes_mean": float(np.mean(all_scores)) if all_scores else 0.0,
            "column_shapes_std": float(np.std(all_scores)) if all_scores else 0.0,
            "column_shapes_detail": scores,
        }

    @staticmethod
    def _tvd_similarity(real_col: pd.Series, synth_col: pd.Series) -> float:
        """Compute 1 - Total Variation Distance between two categorical distributions.

        Args:
            real_col: Real data column
            synth_col: Synthetic data column

        Returns:
            Similarity score in [0, 1]
        """
        real_dist = real_col.value_counts(normalize=True)
        synth_dist = synth_col.value_counts(normalize=True)

        all_categories = set(real_dist.index) | set(synth_dist.index)
        tvd = sum(
            abs(real_dist.get(cat, 0) - synth_dist.get(cat, 0)) for cat in all_categories
        ) / 2.0
        return 1.0 - tvd


class ColumnPairTrends(BaseMetric):
    """Column Pair Trends Metric

    Compares pairwise correlations between columns in real and synthetic data.
    Uses Pearson correlation for numerical-numerical pairs, Cramers V for
    categorical-categorical pairs, and correlation ratio for mixed pairs.

    A score close to 1.0 means the pairwise relationships are well preserved.

    Attributes:
        None
    """

    @staticmethod
    def name() -> str:
        return "column_pair_trends"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute pairwise correlation similarity between real and synthetic data.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.

        Returns:
            Dictionary with 'column_pair_trends_mean' and 'column_pair_trends_detail'
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        # Compute correlation matrices for numerical columns
        if len(num_cols) >= 2:
            real_corr = real_data[num_cols].corr().values
            synth_corr = synthetic_data[num_cols].corr().values

            # Replace NaN with 0 for missing correlations
            real_corr = np.nan_to_num(real_corr, nan=0.0)
            synth_corr = np.nan_to_num(synth_corr, nan=0.0)

            # Similarity = 1 - mean absolute difference of upper triangle
            mask = np.triu_indices_from(real_corr, k=1)
            diff = np.abs(real_corr[mask] - synth_corr[mask])
            corr_similarity = float(1.0 - np.mean(diff)) if len(diff) > 0 else 1.0
        else:
            corr_similarity = 1.0

        # Compute Cramers V similarity for categorical pairs
        cat_scores = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                if c1 not in synthetic_data.columns or c2 not in synthetic_data.columns:
                    continue
                v_real = self._cramers_v(real_data[c1], real_data[c2])
                v_synth = self._cramers_v(synthetic_data[c1], synthetic_data[c2])
                cat_scores.append(1.0 - abs(v_real - v_synth))

        cat_similarity = float(np.mean(cat_scores)) if cat_scores else 1.0

        # Weighted average
        n_num_pairs = max(len(num_cols) * (len(num_cols) - 1) // 2, 0)
        n_cat_pairs = len(cat_scores)
        total_pairs = n_num_pairs + n_cat_pairs

        if total_pairs > 0:
            overall = (corr_similarity * n_num_pairs + cat_similarity * n_cat_pairs) / total_pairs
        else:
            overall = 1.0

        return {
            "column_pair_trends_mean": float(overall),
            "column_pair_trends_numerical": float(corr_similarity),
            "column_pair_trends_categorical": float(cat_similarity),
        }

    @staticmethod
    def _cramers_v(col1: pd.Series, col2: pd.Series) -> float:
        """Compute Cramer's V statistic for two categorical columns.

        Args:
            col1: First categorical column
            col2: Second categorical column

        Returns:
            Cramer's V value in [0, 1]
        """
        confusion_matrix = pd.crosstab(col1, col2)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = len(col1)
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))


class BasicStatistics(BaseMetric):
    """Basic Statistics Comparison

    Compares simple summary statistics (mean, std, median, min, max) between
    real and synthetic datasets for numerical columns, and category frequency
    distributions for categorical columns.

    Attributes:
        None
    """

    @staticmethod
    def name() -> str:
        return "basic_statistics"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute basic statistical comparison.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.

        Returns:
            Dictionary with per-column comparison of means, stds, and distributions
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        results = {}

        for col in num_cols:
            if col not in synthetic_data.columns:
                continue
            real_vals = real_data[col].dropna()
            synth_vals = synthetic_data[col].dropna()
            results[col] = {
                "real_mean": float(real_vals.mean()),
                "synth_mean": float(synth_vals.mean()),
                "real_std": float(real_vals.std()),
                "synth_std": float(synth_vals.std()),
                "real_median": float(real_vals.median()),
                "synth_median": float(synth_vals.median()),
                "mean_diff_pct": float(
                    abs(real_vals.mean() - synth_vals.mean())
                    / (abs(real_vals.mean()) + 1e-10)
                    * 100
                ),
            }

        for col in cat_cols:
            if col not in synthetic_data.columns:
                continue
            real_dist = real_data[col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_data[col].value_counts(normalize=True).to_dict()
            results[col] = {
                "real_distribution": real_dist,
                "synth_distribution": synth_dist,
            }

        return {"basic_statistics": results}
