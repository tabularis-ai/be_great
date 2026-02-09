import logging
import typing as tp

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from be_great.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class MLEfficiency(BaseMetric):
    """Machine Learning Efficiency Metric

    Measures the utility of synthetic data by training a model on the synthetic
    dataset and evaluating it on the real (held-out) test dataset. The closer the
    score is to the performance achieved when training on real data, the higher the
    utility of the synthetic data.

    Attributes:
        model (type): Sklearn-compatible model class (e.g. RandomForestClassifier)
        metric (callable): Scoring function (e.g. accuracy_score, f1_score)
        model_params (dict): Parameters passed to the model constructor
        encoder (type): Encoder class for categorical features
        encoder_params (dict): Parameters for the encoder
        normalize (bool): Whether to standard-scale continuous features
        use_proba (bool): Whether to use predict_proba instead of predict
        metric_params (dict): Extra keyword arguments passed to the scoring function
    """

    def __init__(
        self,
        model: type,
        metric: tp.Callable,
        model_params: tp.Optional[dict] = None,
        encoder: type = OrdinalEncoder,
        encoder_params: tp.Optional[dict] = None,
        normalize: bool = False,
        use_proba: bool = False,
        metric_params: tp.Optional[dict] = None,
    ):
        """Initializes the ML Efficiency Metric.

        Args:
            model: Sklearn-compatible model class
            metric: Scoring function (e.g. accuracy_score)
            model_params: Parameters for the model constructor
            encoder: Encoder class for categorical columns (default: OrdinalEncoder)
            encoder_params: Parameters for the encoder
            normalize: Whether to standard-scale continuous features
            use_proba: Use predict_proba instead of predict for scoring
            metric_params: Additional keyword arguments for the scoring function
        """
        self.model = model
        self.metric = metric
        self.model_params = model_params or {}
        self.encoder = encoder
        self.encoder_params = encoder_params or {
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
        }
        self.normalize = normalize
        self.use_proba = use_proba
        self.metric_params = metric_params or {}

    @staticmethod
    def name() -> str:
        return "ml_efficiency"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        label_col: str = None,
        cat_cols: tp.Optional[tp.List[str]] = None,
        num_cols: tp.Optional[tp.List[str]] = None,
        real_test_data: tp.Optional[pd.DataFrame] = None,
        test_ratio: float = 0.2,
        random_seeds: tp.Optional[tp.List[int]] = None,
    ) -> dict:
        """Compute the ML efficiency score.

        Trains the model on synthetic data and evaluates on real test data.

        Args:
            real_data: The original training dataset (used to fit encoders and as fallback test set)
            synthetic_data: The synthetically generated dataset (used for training)
            label_col: Name of the target/label column
            cat_cols: List of categorical column names. Auto-detected if None.
            num_cols: List of numerical column names. Auto-detected if None.
            real_test_data: Separate real test set. If None, real_data is split using test_ratio.
            test_ratio: Fraction of real_data to use as test set if real_test_data is not provided
            random_seeds: List of random seeds for multiple evaluation runs

        Returns:
            Dictionary with 'mle_mean', 'mle_std', and 'mle_scores'
        """
        assert label_col is not None, "label_col must be specified"
        if random_seeds is None:
            random_seeds = [512, 13, 23, 28, 21]

        feature_cols = [c for c in real_data.columns if c != label_col]
        num_cols, cat_cols = self._detect_column_types(
            real_data[feature_cols], num_cols=num_cols, cat_cols=cat_cols
        )

        # Prepare encoders and test data
        X_test, y_test, cat_encoder, scaler = self._prepare_test(
            real_data, real_test_data, label_col, cat_cols, num_cols, test_ratio
        )

        # Prepare synthetic training data
        X_train, y_train = self._prepare_train(
            synthetic_data, label_col, cat_cols, num_cols, cat_encoder, scaler
        )

        # Evaluate over multiple seeds
        scores = []
        for seed in random_seeds:
            try:
                m = self.model(**self.model_params, random_state=seed)
            except TypeError:
                logger.debug("Model does not accept random_state, using params only")
                m = self.model(**self.model_params)

            m.fit(X_train, y_train)

            if self.use_proba:
                y_pred = m.predict_proba(X_test)[:, 1]
            else:
                y_pred = m.predict(X_test)

            score = self.metric(y_test, y_pred, **self.metric_params)
            scores.append(score)

        return {
            "mle_scores": scores,
            "mle_mean": float(np.mean(scores)),
            "mle_std": float(np.std(scores)),
        }

    def _prepare_test(
        self,
        real_data: pd.DataFrame,
        real_test_data: tp.Optional[pd.DataFrame],
        label_col: str,
        cat_cols: tp.List[str],
        num_cols: tp.List[str],
        test_ratio: float,
    ) -> tp.Tuple[np.ndarray, np.ndarray, tp.Any, tp.Optional[StandardScaler]]:
        """Fit encoders on real training data and prepare the test set.

        Args:
            real_data: Original training data
            real_test_data: Separate test set (if available)
            label_col: Target column name
            cat_cols: Categorical column names
            num_cols: Numerical column names
            test_ratio: Split ratio if no separate test set

        Returns:
            Tuple of (X_test, y_test, fitted_cat_encoder, fitted_scaler)
        """
        df_train = real_data.fillna(0).copy()

        # Fit categorical encoder on training data
        cat_encoder = None
        if cat_cols:
            cat_encoder = self.encoder(**self.encoder_params)
            cat_encoder.fit(df_train[cat_cols])

        # Fit scaler on training data
        scaler = None
        if self.normalize and num_cols:
            scaler = StandardScaler()
            scaler.fit(df_train[num_cols])

        # Get or split test data
        if real_test_data is not None:
            df_test = real_test_data.fillna(0).copy()
        else:
            split_idx = int(len(df_train) * (1 - test_ratio))
            df_test = df_train.iloc[split_idx:].copy()

        # Transform test data
        if cat_encoder is not None and cat_cols:
            df_test[cat_cols] = cat_encoder.transform(df_test[cat_cols])
        if scaler is not None and num_cols:
            df_test[num_cols] = scaler.transform(df_test[num_cols])

        X_test = df_test.drop(label_col, axis=1).to_numpy()
        y_test = df_test[label_col].to_numpy()

        return X_test, y_test, cat_encoder, scaler

    @staticmethod
    def _prepare_train(
        synthetic_data: pd.DataFrame,
        label_col: str,
        cat_cols: tp.List[str],
        num_cols: tp.List[str],
        cat_encoder: tp.Any,
        scaler: tp.Optional[StandardScaler],
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Encode the synthetic training data using pre-fitted encoders.

        Args:
            synthetic_data: Generated dataset
            label_col: Target column name
            cat_cols: Categorical column names
            num_cols: Numerical column names
            cat_encoder: Pre-fitted categorical encoder
            scaler: Pre-fitted standard scaler (or None)

        Returns:
            Tuple of (X_train, y_train) as numpy arrays
        """
        df = synthetic_data.fillna(0).copy().reset_index(drop=True)

        if cat_encoder is not None and cat_cols:
            df[cat_cols] = cat_encoder.transform(df[cat_cols])
        if scaler is not None and num_cols:
            df[num_cols] = scaler.transform(df[num_cols])

        X_train = df.drop(label_col, axis=1).to_numpy()
        y_train = df[label_col].to_numpy()

        return X_train, y_train
