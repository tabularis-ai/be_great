import logging
import typing as tp

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder

from be_great.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class DiscriminatorMetric(BaseMetric):
    """Discriminator Metric

    Trains a classifier to distinguish between real and synthetic data.
    A score close to 0.5 indicates the synthetic data is indistinguishable
    from the real data. A score close to 1.0 means the classifier can
    easily tell them apart.

    The metric uses a cross-validated Random Forest and reports mean and
    standard deviation over multiple runs with different random seeds.

    Attributes:
        metric (callable): Scoring function (e.g. accuracy_score)
        n_runs (int): Number of evaluation runs with different seeds
        encoder (type): Encoder class for categorical features
        encoder_params (dict): Parameters for the encoder
    """

    def __init__(
        self,
        metric: tp.Callable = accuracy_score,
        n_runs: int = 10,
        encoder: type = OrdinalEncoder,
        encoder_params: tp.Optional[dict] = None,
    ):
        """Initializes the Discriminator Metric.

        Args:
            metric: Sklearn-compatible scoring function (default: accuracy_score)
            n_runs: Number of evaluation runs with different random seeds
            encoder: Encoder class for categorical columns (default: OrdinalEncoder)
            encoder_params: Parameters passed to the encoder
        """
        self.metric = metric
        self.n_runs = n_runs
        self.encoder = encoder
        self.encoder_params = encoder_params or {
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
        }

    @staticmethod
    def name() -> str:
        return "discriminator"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        cat_cols: tp.Optional[tp.List[str]] = None,
        test_ratio: float = 0.2,
        cv: int = 5,
    ) -> dict:
        """Compute the discriminator score.

        Args:
            real_data: The original tabular dataset
            synthetic_data: The synthetically generated dataset
            cat_cols: List of categorical column names. Auto-detected if None.
            test_ratio: Fraction of data used for testing
            cv: Number of cross-validation folds for hyperparameter tuning

        Returns:
            Dictionary with 'discriminator_mean' and 'discriminator_std'
        """
        _, cat_cols = self._detect_column_types(real_data, cat_cols=cat_cols)

        # Encode categorical features
        X_real, X_synth = self._encode(real_data, synthetic_data, cat_cols)

        # Build discriminator dataset (real=0, synthetic=1)
        X_train, X_test, y_train, y_test = self._build_disc_split(
            X_real, X_synth, test_ratio
        )

        # Tune hyperparameters via grid search
        best_params = self._tune_classifier(X_train, y_train, cv)

        # Evaluate over multiple runs
        scores = self._evaluate_runs(X_train, y_train, X_test, y_test, best_params)

        return {
            "discriminator_mean": float(np.mean(scores)),
            "discriminator_std": float(np.std(scores)),
        }

    def _encode(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        cat_cols: tp.List[str],
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Fit encoder on real data and transform both datasets.

        Args:
            real_data: Original data used to fit the encoder
            synthetic_data: Generated data to transform
            cat_cols: Categorical column names

        Returns:
            Tuple of encoded (real, synthetic) numpy arrays
        """
        df_real = real_data.copy()
        df_synth = synthetic_data.copy()

        if cat_cols:
            cat_encoder = self.encoder(**self.encoder_params)
            cat_encoder.fit(df_real[cat_cols])
            df_real[cat_cols] = cat_encoder.transform(df_real[cat_cols])
            df_synth[cat_cols] = cat_encoder.transform(
                df_synth[cat_cols].reindex(columns=df_real[cat_cols].columns)
            )

        # Ensure same column order
        df_synth = df_synth[df_real.columns]
        return df_real.to_numpy(), df_synth.to_numpy()

    @staticmethod
    def _build_disc_split(
        X_real: np.ndarray,
        X_synth: np.ndarray,
        test_ratio: float,
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create balanced train/test split for the discriminator.

        Args:
            X_real: Encoded real data
            X_synth: Encoded synthetic data
            test_ratio: Fraction used for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        n_use = min(len(X_real), len(X_synth))
        X_real = X_real[:n_use]
        X_synth = X_synth[:n_use]

        split_idx = int(n_use * (1 - test_ratio))

        X_train = np.r_[X_real[:split_idx], X_synth[:split_idx]]
        y_train = np.r_[np.zeros(split_idx), np.ones(split_idx)]

        X_test = np.r_[X_real[split_idx:], X_synth[split_idx:]]
        y_test = np.r_[np.zeros(n_use - split_idx), np.ones(n_use - split_idx)]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _tune_classifier(
        X_train: np.ndarray, y_train: np.ndarray, cv: int
    ) -> dict:
        """Tune Random Forest hyperparameters via grid search.

        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of CV folds

        Returns:
            Best parameter dictionary
        """
        param_grid = {
            "n_estimators": [100],
            "max_features": ["sqrt"],
            "max_depth": [10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [2, 3, 4],
            "bootstrap": [True, False],
        }

        search = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        logger.info("Best discriminator params: %s", search.best_params_)
        return search.best_params_

    def _evaluate_runs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        best_params: dict,
    ) -> tp.List[float]:
        """Evaluate the discriminator over multiple random seeds.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            best_params: Tuned hyperparameters

        Returns:
            List of metric scores, one per run
        """
        scores = []
        for i in range(self.n_runs):
            params = {**best_params, "random_state": i * 42}
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores.append(self.metric(y_test, pred))
        return scores
