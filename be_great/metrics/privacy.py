import logging
import typing as tp
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from be_great.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class DistanceToClosestRecord(BaseMetric):
    """Distance to Closest Record (DCR)

    For each synthetic record, computes the distance to the closest record in the
    real training dataset. Uses L1 (Manhattan) distance for numerical features and
    Hamming distance (0/1 mismatch) for categorical features.

    A high average distance indicates the model is not simply memorizing training
    data. Records with distance 0 are exact copies.

    Attributes:
        n_samples (int): Number of synthetic samples to evaluate (0 = use all)
        use_euclidean (bool): Use L2 instead of L1 distance for numerical features
    """

    def __init__(self, n_samples: int = 0, use_euclidean: bool = False):
        """Initializes the DCR Metric.

        Args:
            n_samples: Number of synthetic samples to evaluate. 0 means use all.
            use_euclidean: If True, use L2 norm for numerical features instead of L1
        """
        self.n_samples = n_samples
        self.use_euclidean = use_euclidean

    @staticmethod
    def name() -> str:
        return "distance_to_closest_record"

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
        """Compute the DCR metric.

        Args:
            real_data: The original training dataset
            synthetic_data: The synthetically generated dataset
            num_cols: List of numerical column names. Auto-detected if None.
            cat_cols: List of categorical column names. Auto-detected if None.

        Returns:
            Dictionary with 'dcr_mean', 'dcr_std', 'dcr_min', 'n_copies',
            'ratio_copies', and 'distances' (full list)
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        df_gen = synthetic_data
        if self.n_samples > 0 and self.n_samples < len(synthetic_data):
            df_gen = synthetic_data.sample(n=self.n_samples, random_state=42)

        if self.use_euclidean:
            distances = self._compute_l2(real_data, df_gen, num_cols, cat_cols)
        else:
            distances = self._compute_l1(real_data, df_gen, num_cols, cat_cols)

        n_copies = sum(d == 0 for d in distances)
        distances_arr = np.array(distances)

        return {
            "dcr_mean": float(distances_arr.mean()),
            "dcr_std": float(distances_arr.std()),
            "dcr_min": float(distances_arr.min()),
            "n_copies": int(n_copies),
            "ratio_copies": float(n_copies / len(distances)),
            "distances": distances,
        }

    @staticmethod
    def _compute_l1(
        df_orig: pd.DataFrame,
        df_gen: pd.DataFrame,
        num_cols: tp.List[str],
        cat_cols: tp.List[str],
    ) -> tp.List[float]:
        """Compute minimum L1 distance for each synthetic record.

        Args:
            df_orig: Real dataset
            df_gen: Synthetic dataset
            num_cols: Numerical column names
            cat_cols: Categorical column names

        Returns:
            List of minimum distances, one per synthetic sample
        """
        mins = []
        for i in range(len(df_gen)):
            sample = df_gen.iloc[i]
            dist_num = abs(df_orig[num_cols] - sample[num_cols]).sum(axis=1) if num_cols else 0
            dist_cat = (~(df_orig[cat_cols] == sample[cat_cols])).sum(axis=1) if cat_cols else 0
            min_total = (dist_num + dist_cat).min()
            mins.append(float(min_total))
        return mins

    @staticmethod
    def _compute_l2(
        df_orig: pd.DataFrame,
        df_gen: pd.DataFrame,
        num_cols: tp.List[str],
        cat_cols: tp.List[str],
    ) -> tp.List[float]:
        """Compute minimum L2 distance for each synthetic record.

        Args:
            df_orig: Real dataset
            df_gen: Synthetic dataset
            num_cols: Numerical column names
            cat_cols: Categorical column names

        Returns:
            List of minimum distances, one per synthetic sample
        """
        mins = []
        for i in range(len(df_gen)):
            sample = df_gen.iloc[i]
            dist_num = (
                np.sqrt(((df_orig[num_cols] - sample[num_cols]) ** 2).sum(axis=1))
                if num_cols else 0
            )
            dist_cat = (~(df_orig[cat_cols] == sample[cat_cols])).sum(axis=1) if cat_cols else 0
            min_total = (dist_num + dist_cat).min()
            mins.append(float(min_total))
        return mins


class kAnonymization(BaseMetric):
    """k-Anonymization Metric

    Evaluates the k-anonymity of a dataset. Each record should be similar to at
    least k-1 other records on the quasi-identifying variables. Uses KMeans
    clustering to estimate the minimum group size.

    Reports the ratio syn_k / real_k. A ratio >= 1 means the synthetic data
    has at least as much k-anonymity as the real data.

    Attributes:
        n_clusters_list (list[int]): List of cluster counts to evaluate
    """

    def __init__(self, n_clusters_list: tp.Optional[tp.List[int]] = None):
        """Initializes the k-Anonymization Metric.

        Args:
            n_clusters_list: List of cluster sizes to try. Defaults to [2, 5, 10, 15].
        """
        self.n_clusters_list = n_clusters_list or [2, 5, 10, 15]

    @staticmethod
    def name() -> str:
        return "k_anonymization"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute k-anonymization for both real and synthetic data.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            sensitive_cols: Columns to exclude from quasi-identifiers.
                If None, all columns are used.

        Returns:
            Dictionary with 'k_real', 'k_synthetic', and 'k_ratio'
        """
        feature_cols = self._get_feature_cols(real_data, sensitive_cols)

        k_real = self._estimate_k(real_data[feature_cols])
        k_synth = self._estimate_k(synthetic_data[feature_cols])

        return {
            "k_real": int(k_real),
            "k_synthetic": int(k_synth),
            "k_ratio": float((k_synth + 1e-8) / (k_real + 1e-8)),
        }

    def _estimate_k(self, df: pd.DataFrame) -> int:
        """Estimate the k-anonymity value using KMeans clustering.

        Args:
            df: DataFrame with quasi-identifier columns (numeric-encoded)

        Returns:
            Minimum cluster size across all tested cluster counts
        """
        # Encode categoricals for clustering
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes

        values = [999]
        for n_clusters in self.n_clusters_list:
            if len(df_encoded) / n_clusters < 10:
                continue
            kmeans = KMeans(
                n_clusters=n_clusters, init="k-means++", random_state=0, n_init=10
            )
            labels = kmeans.fit_predict(df_encoded)
            counts = Counter(labels)
            values.append(min(counts.values()))

        return int(np.min(values))

    @staticmethod
    def _get_feature_cols(
        df: pd.DataFrame, sensitive_cols: tp.Optional[tp.List[str]]
    ) -> tp.List[str]:
        """Get quasi-identifier columns by excluding sensitive ones.

        Args:
            df: DataFrame to inspect
            sensitive_cols: Columns to exclude

        Returns:
            List of quasi-identifier column names
        """
        if sensitive_cols is None:
            return list(df.columns)
        return [c for c in df.columns if c not in sensitive_cols]


class lDiversity(BaseMetric):
    """l-Diversity Metric

    Measures the diversity of sensitive attribute values within each equivalence
    class. Uses KMeans to form groups and checks how many distinct sensitive
    values exist in the smallest group.

    Higher l-diversity means better privacy protection against attribute inference.

    Attributes:
        sensitive_col (str): Name of the sensitive column to evaluate
        n_clusters_list (list[int]): List of cluster counts to evaluate
    """

    def __init__(
        self,
        sensitive_col: str,
        n_clusters_list: tp.Optional[tp.List[int]] = None,
    ):
        """Initializes the l-Diversity Metric.

        Args:
            sensitive_col: Name of the sensitive attribute column
            n_clusters_list: List of cluster sizes to try. Defaults to [2, 5, 10, 15].
        """
        self.sensitive_col = sensitive_col
        self.n_clusters_list = n_clusters_list or [2, 5, 10, 15]

    @staticmethod
    def name() -> str:
        return "l_diversity"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict:
        """Compute l-diversity for both real and synthetic datasets.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset

        Returns:
            Dictionary with 'l_real', 'l_synthetic', and 'l_ratio'
        """
        l_real = self._estimate_l(real_data)
        l_synth = self._estimate_l(synthetic_data)

        return {
            "l_real": int(l_real),
            "l_synthetic": int(l_synth),
            "l_ratio": float((l_synth + 1e-8) / (l_real + 1e-8)),
        }

    def _estimate_l(self, df: pd.DataFrame) -> int:
        """Estimate l-diversity using KMeans equivalence classes.

        Args:
            df: Full dataset including the sensitive column

        Returns:
            Minimum number of distinct sensitive values across all groups
        """
        quasi_cols = [c for c in df.columns if c != self.sensitive_col]

        # Encode for clustering
        df_encoded = df[quasi_cols].copy()
        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes

        min_diversity = len(df[self.sensitive_col].unique())

        for n_clusters in self.n_clusters_list:
            if len(df_encoded) / n_clusters < 10:
                continue
            kmeans = KMeans(
                n_clusters=n_clusters, init="k-means++", random_state=0, n_init=10
            )
            labels = kmeans.fit_predict(df_encoded)

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                n_unique = df.loc[mask, self.sensitive_col].nunique()
                min_diversity = min(min_diversity, n_unique)

        return int(min_diversity)


class IdentifiabilityScore(BaseMetric):
    """Identifiability Score

    Measures the risk that a synthetic record can be linked back to a specific
    real record. Uses k-nearest neighbors to find, for each synthetic record,
    whether its closest real neighbor is uniquely identifiable (i.e., significantly
    closer than the second closest).

    A lower score means better privacy.

    Attributes:
        n_neighbors (int): Number of neighbors to consider
        threshold_ratio (float): Distance ratio threshold for identifiability
    """

    def __init__(self, n_neighbors: int = 5, threshold_ratio: float = 0.5):
        """Initializes the Identifiability Score.

        Args:
            n_neighbors: Number of nearest neighbors to compute
            threshold_ratio: A synthetic record is considered identifiable if
                distance_to_1st / distance_to_2nd < threshold_ratio
        """
        self.n_neighbors = n_neighbors
        self.threshold_ratio = threshold_ratio

    @staticmethod
    def name() -> str:
        return "identifiability_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute the identifiability score.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.

        Returns:
            Dictionary with 'identifiability_score' (fraction of identifiable records)
            and 'mean_distance_ratio'
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        # Encode all data to numeric
        real_enc = self._encode_for_nn(real_data, num_cols, cat_cols)
        synth_enc = self._encode_for_nn(synthetic_data, num_cols, cat_cols)

        k = min(self.n_neighbors, len(real_enc) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2)
        nn.fit(real_enc)
        distances, _ = nn.kneighbors(synth_enc)

        # Compute identifiability ratio: dist_1st / dist_2nd
        n_identifiable = 0
        ratios = []
        for i in range(len(distances)):
            d1 = distances[i, 0]
            d2 = distances[i, 1] if k >= 2 else d1
            ratio = d1 / (d2 + 1e-10)
            ratios.append(ratio)
            if ratio < self.threshold_ratio:
                n_identifiable += 1

        return {
            "identifiability_score": float(n_identifiable / len(distances)),
            "mean_distance_ratio": float(np.mean(ratios)),
        }

    @staticmethod
    def _encode_for_nn(
        df: pd.DataFrame, num_cols: tp.List[str], cat_cols: tp.List[str]
    ) -> np.ndarray:
        """Encode DataFrame to a numeric array suitable for nearest neighbors.

        Args:
            df: DataFrame to encode
            num_cols: Numerical columns (used as-is)
            cat_cols: Categorical columns (encoded to category codes)

        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        df_enc = df.copy()
        for col in cat_cols:
            df_enc[col] = df_enc[col].astype("category").cat.codes.astype(float)
        cols = num_cols + cat_cols
        return df_enc[cols].fillna(0).to_numpy(dtype=float)


class DeltaPresence(BaseMetric):
    """Delta-Presence Metric

    Measures how much the presence of an individual in the dataset can be inferred
    from the synthetic data. Computes the fraction of real records that have a
    near-exact match in the synthetic dataset within a distance threshold.

    Lower delta-presence means better privacy.

    Attributes:
        threshold (float): Distance threshold below which a record is considered "present"
    """

    def __init__(self, threshold: float = 0.0):
        """Initializes the Delta-Presence Metric.

        Args:
            threshold: Distance threshold. A real record is "present" in the
                synthetic data if its nearest synthetic neighbor is within this distance.
                Default 0.0 means exact match only.
        """
        self.threshold = threshold

    @staticmethod
    def name() -> str:
        return "delta_presence"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
    ) -> dict:
        """Compute the delta-presence metric.

        Args:
            real_data: The original dataset
            synthetic_data: The synthetically generated dataset
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.

        Returns:
            Dictionary with 'delta_presence' (fraction of real records with a match),
            and 'mean_nearest_distance'
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        real_enc = IdentifiabilityScore._encode_for_nn(real_data, num_cols, cat_cols)
        synth_enc = IdentifiabilityScore._encode_for_nn(synthetic_data, num_cols, cat_cols)

        nn = NearestNeighbors(n_neighbors=1, metric="minkowski", p=2)
        nn.fit(synth_enc)
        distances, _ = nn.kneighbors(real_enc)
        distances = distances.flatten()

        n_present = int(np.sum(distances <= self.threshold))

        return {
            "delta_presence": float(n_present / len(distances)),
            "mean_nearest_distance": float(distances.mean()),
        }


class MembershipInference(BaseMetric):
    """Membership Inference Risk

    Simulates a membership inference attack: given a record, can an attacker
    determine whether it was in the training set? This is estimated by comparing
    distances from known-member records (train) and known-non-member records (holdout)
    to their nearest synthetic neighbors, then measuring how separable the two
    groups are.

    A score close to 0.5 means the attacker cannot distinguish members from non-members
    (good privacy). A score close to 1.0 means high membership inference risk.

    Attributes:
        n_neighbors (int): Number of neighbors for the distance computation
    """

    def __init__(self, n_neighbors: int = 1):
        """Initializes the Membership Inference Metric.

        Args:
            n_neighbors: Number of nearest neighbors to use
        """
        self.n_neighbors = n_neighbors

    @staticmethod
    def name() -> str:
        return "membership_inference"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        holdout_data: tp.Optional[pd.DataFrame] = None,
        num_cols: tp.Optional[tp.List[str]] = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
        holdout_ratio: float = 0.5,
    ) -> dict:
        """Compute membership inference risk.

        Args:
            real_data: The original training dataset (members)
            synthetic_data: The synthetically generated dataset
            holdout_data: Non-member data. If None, real_data is split using holdout_ratio.
            num_cols: Numerical column names. Auto-detected if None.
            cat_cols: Categorical column names. Auto-detected if None.
            holdout_ratio: Fraction to split off as holdout if holdout_data is not provided

        Returns:
            Dictionary with 'membership_inference_score' (attacker accuracy)
        """
        num_cols, cat_cols = self._detect_column_types(
            real_data, num_cols=num_cols, cat_cols=cat_cols
        )

        if holdout_data is None:
            split_idx = int(len(real_data) * (1 - holdout_ratio))
            member_data = real_data.iloc[:split_idx]
            non_member_data = real_data.iloc[split_idx:]
        else:
            member_data = real_data
            non_member_data = holdout_data

        synth_enc = IdentifiabilityScore._encode_for_nn(
            synthetic_data, num_cols, cat_cols
        )
        member_enc = IdentifiabilityScore._encode_for_nn(
            member_data, num_cols, cat_cols
        )
        non_member_enc = IdentifiabilityScore._encode_for_nn(
            non_member_data, num_cols, cat_cols
        )

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="minkowski", p=2)
        nn.fit(synth_enc)

        member_dists = nn.kneighbors(member_enc)[0].mean(axis=1)
        non_member_dists = nn.kneighbors(non_member_enc)[0].mean(axis=1)

        # Use median distance as threshold for a simple attack
        all_dists = np.concatenate([member_dists, non_member_dists])
        threshold = np.median(all_dists)

        # Members should have smaller distances to synthetic data
        member_correct = np.sum(member_dists <= threshold)
        non_member_correct = np.sum(non_member_dists > threshold)
        total = len(member_dists) + len(non_member_dists)

        accuracy = (member_correct + non_member_correct) / total

        return {
            "membership_inference_score": float(accuracy),
            "mean_member_distance": float(member_dists.mean()),
            "mean_non_member_distance": float(non_member_dists.mean()),
        }
