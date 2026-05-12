"""Schema-driven mock data generation for GReaT.

This module supports ``GReaT.mock(...)`` — generating synthetic tabular data
**without fitting on real data**. Instead of learning a distribution from a
training set, the user declares a *schema* describing each column. GReaT
populates the same internal state that ``fit()`` normally builds (columns,
num_cols, col_stats, conditional_col_dist) directly from that schema, then
runs the existing guided/constrained sampling pipeline.

The schema is a ``Dict[str, Dict[str, Any]]`` mapping column name to a spec.

Numerical column spec
---------------------

.. code-block:: python

    "age": {
        "type": "num",
        "range": (18, 99),           # required: (min, max)
        "integer": True,             # optional: enumerate whole numbers only
        "precision": 2,              # optional: decimal places (overrides 'integer')
        "dist": "normal",            # optional: distribution hint
        "mean": 38, "std": 13,       # required when dist == "normal"
    }

Categorical column spec
-----------------------

.. code-block:: python

    "sex": {
        "type": "cat",
        "values": ["Male", "Female", "Other"],   # required
        "weights": [0.66, 0.33, 0.01],           # optional: per-value prior
        # weights may also be a dict: {"Male": 0.66, ...}
    }

Conditions
----------

The standard ``sample()`` ``conditions=`` syntax works:

.. code-block:: python

    conditions={"age": ">= 40", "sex": "!= 'Male'"}

Schema declarations become *implicit* constraints — every schema column is
automatically constrained to its declared range/values, even when the user
doesn't list it in ``conditions``. User-supplied conditions tighten those
defaults.
"""

import json
import os
import random
import typing as tp

import numpy as np
import pandas as pd


# Module-level constants for schema keys. Use these instead of string literals
# in caller code to keep things DRY.
TYPE_NUM = ("num", "numeric", "numerical", "number")
TYPE_CAT = ("cat", "categorical", "category")

# Sentinel string used to build a match-all constraint on a categorical column
# whose values are declared in the schema but for which the user did not
# provide an explicit condition. The trie includes all categories that are
# ``!=`` this sentinel — i.e. all of them, as long as no real category equals
# this string.
_CAT_NO_MATCH_SENTINEL = "__GREAT_NO_MATCH__"


def populate_schema_state(
    model,
    schema: tp.Dict[str, tp.Dict[str, tp.Any]],
    conditional_col: tp.Optional[str] = None,
) -> None:
    """Populate a ``GReaT`` instance's state from a declarative schema.

    Sets ``model.columns``, ``model.num_cols``, ``model.col_stats``,
    ``model.conditional_col``, ``model.conditional_col_dist`` — the same
    attributes ``fit()`` populates from real data. The conditional column
    distribution is uniform (no real data to draw from).

    Args:
        model: A ``GReaT`` instance.
        schema: Schema dict. See module docstring for format.
        conditional_col: Column to use as the generation starting point.
            Defaults to the *last* column in the schema, matching ``fit()``.

    Raises:
        ValueError: If the schema is empty or malformed.
    """
    if not schema:
        raise ValueError("schema cannot be empty")

    columns: tp.List[str] = list(schema.keys())
    num_cols: tp.List[str] = []
    col_stats: tp.Dict[str, tp.Dict[str, tp.Any]] = {}

    for col, spec in schema.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"schema[{col!r}] must be a dict, got {type(spec).__name__}"
            )
        ctype = str(spec.get("type", "")).lower()
        if ctype in TYPE_NUM:
            col_stats[col] = _parse_numeric_spec(col, spec)
            num_cols.append(col)
        elif ctype in TYPE_CAT:
            col_stats[col] = _parse_categorical_spec(col, spec)
        else:
            raise ValueError(
                f"Column {col!r}: type must be one of {TYPE_NUM + TYPE_CAT}, got {ctype!r}"
            )

    if conditional_col is None:
        conditional_col = columns[-1]
    if conditional_col not in columns:
        raise ValueError(
            f"conditional_col={conditional_col!r} not in schema columns {columns}"
        )

    model.columns = columns
    model.num_cols = num_cols
    model.col_stats = col_stats
    model.conditional_col = conditional_col
    model.conditional_col_dist = _uniform_conditional_dist(col_stats[conditional_col])


def build_effective_conditions(
    col_stats: tp.Dict[str, tp.Dict[str, tp.Any]],
    user_conditions: tp.Optional[tp.Dict[str, str]],
) -> tp.Dict[str, str]:
    """Compose user conditions with implicit schema-derived constraints.

    Every schema column gets an implicit constraint so the trie/processor
    enforces the declared range or value set. User conditions, when provided,
    tighten the defaults.

    Args:
        col_stats: ``model.col_stats`` populated by ``populate_schema_state``.
        user_conditions: Optional ``{col: condition_str}`` from the caller.

    Returns:
        Merged conditions dict ready to pass to ``_guided_sample``.

    Raises:
        ValueError: If a user condition references a column not in the schema.
    """
    if user_conditions:
        for col in user_conditions:
            if col not in col_stats:
                raise ValueError(
                    f"Condition column {col!r} not in schema {list(col_stats.keys())}"
                )

    effective: tp.Dict[str, str] = {}
    for col, stats in col_stats.items():
        if stats["type"] == "categorical":
            effective[col] = f"!= '{_CAT_NO_MATCH_SENTINEL}'"
        else:
            effective[col] = f">= {stats['min']}"
    if user_conditions:
        effective.update(user_conditions)
    return effective


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_numeric_spec(col: str, spec: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    if "range" not in spec:
        raise ValueError(f"Numerical column {col!r} requires 'range': (min, max)")
    lo, hi = spec["range"]
    if float(lo) > float(hi):
        raise ValueError(f"Column {col!r} range min ({lo}) > max ({hi})")

    stats: tp.Dict[str, tp.Any] = {
        "type": "numeric",
        "min": float(lo),
        "max": float(hi),
    }
    if spec.get("integer"):
        stats["precision"] = 0
    elif "precision" in spec:
        stats["precision"] = int(spec["precision"])

    if spec.get("dist") == "normal":
        if "mean" not in spec or "std" not in spec:
            raise ValueError(
                f"Column {col!r} dist='normal' requires 'mean' and 'std'"
            )
        stats["dist"] = "normal"
        stats["mean"] = float(spec["mean"])
        stats["std"] = float(spec["std"])

    _apply_null_prob_to_stats(col, spec, stats)
    return stats


def _parse_categorical_spec(col: str, spec: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    if "values" not in spec or not spec["values"]:
        raise ValueError(
            f"Categorical column {col!r} requires non-empty 'values': [...]"
        )
    cats = [str(v) for v in spec["values"]]
    stats: tp.Dict[str, tp.Any] = {"type": "categorical", "categories": cats}

    if "weights" in spec:
        w = spec["weights"]
        if isinstance(w, dict):
            stats["weights"] = {str(k): float(v) for k, v in w.items()}
        elif isinstance(w, (list, tuple)):
            if len(w) != len(cats):
                raise ValueError(
                    f"Column {col!r}: weights list length {len(w)} does not "
                    f"match values length {len(cats)}"
                )
            stats["weights"] = {c: float(wt) for c, wt in zip(cats, w)}
        else:
            raise ValueError(f"Column {col!r}: 'weights' must be dict or list")

    _apply_null_prob_to_stats(col, spec, stats)
    return stats


def _uniform_conditional_dist(
    cc_stats: tp.Dict[str, tp.Any],
) -> tp.Union[tp.Dict[str, float], tp.List[float]]:
    """Build a uniform start-distribution for the conditional column."""
    if cc_stats["type"] == "categorical":
        cats = cc_stats["categories"]
        return {c: 1.0 / len(cats) for c in cats}
    # Numerical: discretize the range uniformly for the start sampler
    lo, hi = cc_stats["min"], cc_stats["max"]
    return np.linspace(lo, hi, 50).tolist()


def _apply_null_prob_to_stats(
    col: str, spec: tp.Dict[str, tp.Any], stats: tp.Dict[str, tp.Any]
) -> None:
    if "null_prob" not in spec:
        return
    p = float(spec["null_prob"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Column {col!r}: null_prob must be in [0, 1], got {p}")
    if p > 0:
        stats["null_prob"] = p


# ---------------------------------------------------------------------------
# Public helpers used by GReaT.mock()
# ---------------------------------------------------------------------------


def apply_null_probabilities(
    df: pd.DataFrame,
    col_stats: tp.Dict[str, tp.Dict[str, tp.Any]],
    rng: tp.Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Replace cell values with NaN per each column's ``null_prob``.

    Operates on a *copy* of ``df``. Columns without ``null_prob`` are
    untouched. Numerical columns receive ``np.nan``; categorical columns
    receive ``None`` (which pandas renders as NaN in object dtype).

    Args:
        df: DataFrame returned by guided sampling.
        col_stats: Populated by ``populate_schema_state``.
        rng: Optional ``np.random.Generator`` for reproducibility.

    Returns:
        New DataFrame with NaNs randomly inserted.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = df.copy()
    for col, stats in col_stats.items():
        p = stats.get("null_prob")
        if not p or col not in out.columns:
            continue
        mask = rng.random(len(out)) < p
        fill = np.nan if stats["type"] == "numeric" else None
        out.loc[mask, col] = fill
    return out


def build_few_shot_prefix(
    examples: tp.Sequence[tp.Mapping[str, tp.Any]],
    columns: tp.Sequence[str],
) -> str:
    """Format example rows as a GReaT-style prompt prefix.

    Each example becomes ``"col1 is val1, col2 is val2; "`` so the model
    sees a few complete rows before being asked to continue with its own.
    Useful for non-tabular-pretrained LLMs that don't know the GReaT format
    by default.

    Args:
        examples: Iterable of dicts; each dict maps column names to values.
            Missing columns in an example are skipped silently.
        columns: Canonical column ordering (from the schema). Determines
            which keys to emit and in what order.

    Returns:
        A single string ending in ``"; "``; empty string if no usable rows.
    """
    if not examples:
        return ""
    chunks: tp.List[str] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        parts = []
        for col in columns:
            if col in ex and ex[col] is not None:
                parts.append(f"{col} is {ex[col]}")
        if parts:
            chunks.append(", ".join(parts))
    if not chunks:
        return ""
    return "; ".join(chunks) + "; "


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python ``random``, NumPy, and PyTorch for reproducible mock runs.

    Returns a fresh ``np.random.Generator`` seeded from the same value, so
    callers needing local stochasticity (e.g. null masking) get a reusable
    independent stream.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Schema serialization
# ---------------------------------------------------------------------------


def save_schema(schema: tp.Dict[str, tp.Any], path: str) -> None:
    """Write a schema to disk as JSON or YAML (auto-detected by extension).

    Tuples in ``range`` are serialized as JSON lists; ``load_schema`` reads
    them back as lists, which the spec parser accepts transparently.

    Args:
        schema: Schema dict (same format accepted by ``model.mock()``).
        path: Destination path. ``.json`` → JSON; ``.yaml``/``.yml`` → YAML.

    Raises:
        ValueError: For unknown extensions.
        ImportError: If YAML is requested but ``PyYAML`` is not installed.
    """
    ext = os.path.splitext(path)[1].lower()
    serializable = _normalize_schema_for_serialization(schema)
    if ext == ".json":
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Saving schemas as YAML requires PyYAML. Install with: pip install pyyaml"
            ) from e
        with open(path, "w") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)
    else:
        raise ValueError(
            f"Unknown schema file extension {ext!r}. Use .json, .yaml, or .yml."
        )


def load_schema(path: str) -> tp.Dict[str, tp.Any]:
    """Load a schema previously written by ``save_schema``.

    Args:
        path: Path to a ``.json``, ``.yaml``, or ``.yml`` file.

    Returns:
        Schema dict ready to pass to ``model.mock(schema=…)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Loading YAML schemas requires PyYAML. Install with: pip install pyyaml"
            ) from e
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unknown schema file extension {ext!r}. Use .json, .yaml, or .yml."
        )


def _normalize_schema_for_serialization(
    schema: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    out: tp.Dict[str, tp.Any] = {}
    for col, spec in schema.items():
        if not isinstance(spec, dict):
            out[col] = spec
            continue
        clean = dict(spec)
        if "range" in clean and isinstance(clean["range"], tuple):
            clean["range"] = list(clean["range"])
        if "weights" in clean and isinstance(clean["weights"], tuple):
            clean["weights"] = list(clean["weights"])
        out[col] = clean
    return out
