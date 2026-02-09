"""Constrained decoding support for GReaT sampling.

Provides a LogitsProcessor that enforces logical constraints (e.g. ">= 30",
"!= 'New York'") during token generation by walking a prefix trie of valid
value token sequences.
"""

import re
import logging
import typing as tp

import numpy as np
import torch
from transformers import LogitsProcessor


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

# Matches: operator  value
# Examples: ">= 30", "< 50000", "!= 'New York'", "== 'yes'"
_CONDITION_RE = re.compile(
    r"^\s*(>=|<=|!=|==|>|<)\s*"   # operator
    r"('([^']*)'|\"([^\"]*)\"|(.+?))"  # quoted string or bare value
    r"\s*$"
)

_NUMERIC_OPS = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,
    "<":  lambda v, t: v < t,
    "!=": lambda v, t: v != t,
    "==": lambda v, t: v == t,
}

_STRING_OPS = {
    "!=": lambda v, t: v != t,
    "==": lambda v, t: v == t,
}


def parse_condition(condition_str: str) -> tp.Tuple[str, tp.Union[float, str]]:
    """Parse a condition string into (operator, threshold).

    Args:
        condition_str: e.g. ``">= 30"``, ``"!= 'New York'"``

    Returns:
        Tuple of (operator_string, threshold_value).
        threshold_value is float for numeric, str for quoted strings.

    Raises:
        ValueError: If the condition string cannot be parsed.
    """
    m = _CONDITION_RE.match(condition_str)
    if not m:
        raise ValueError(
            f"Cannot parse condition: {condition_str!r}. "
            "Expected format like \">= 30\", \"!= 'New York'\", \"< 50000\"."
        )
    op = m.group(1)
    # Groups 3,4 are quoted content, group 5 is bare value
    value_str = m.group(3) or m.group(4) or m.group(5)
    value_str = value_str.strip()

    # Try numeric conversion
    try:
        threshold = float(value_str)
    except ValueError:
        threshold = value_str

    return op, threshold


# ---------------------------------------------------------------------------
# Numeric formatting (must match GReaTDataset._format_value)
# ---------------------------------------------------------------------------

def _format_numeric(value: float, precision: tp.Optional[int]) -> str:
    """Format a numeric value the same way GReaTDataset._format_value does.

    Args:
        value: The numeric value.
        precision: Number of decimal places (None = full precision).

    Returns:
        Formatted string.
    """
    if precision is not None:
        s = f"{value:.{precision}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    # Full precision: use repr-like formatting but strip trailing zeros
    if value == int(value):
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Enumerate valid values
# ---------------------------------------------------------------------------

_MAX_ENUM_VALUES = 50_000


def _infer_precision(col_min: float, col_max: float) -> int:
    """Infer decimal precision from column range when float_precision is None."""
    col_range = col_max - col_min
    if col_range == 0:
        return 0
    if col_range > 10000:
        return 0
    if col_range > 1000:
        return 1
    if col_range > 10:
        return 2
    if col_range > 1:
        return 3
    return 4


def enumerate_valid_values(
    col_name: str,
    op: str,
    threshold: tp.Union[float, str],
    col_stats: dict,
    float_precision: tp.Optional[int],
) -> tp.List[str]:
    """Enumerate all valid value strings for a constrained column.

    Args:
        col_name: Column name (for error messages).
        op: Comparison operator string.
        threshold: Threshold value.
        col_stats: Dict with keys ``"type"`` (``"numeric"`` or ``"categorical"``),
            and either ``"min"``/``"max"`` or ``"categories"``.
        float_precision: Decimal places used during training.

    Returns:
        List of valid value strings formatted to match training data.

    Raises:
        ValueError: If no valid values exist or the operator is unsupported.
    """
    if col_stats["type"] == "categorical":
        return _enumerate_categorical(col_name, op, threshold, col_stats)
    else:
        return _enumerate_numeric(col_name, op, threshold, col_stats, float_precision)


def _enumerate_categorical(
    col_name: str,
    op: str,
    threshold: tp.Union[float, str],
    col_stats: dict,
) -> tp.List[str]:
    """Enumerate valid categorical values."""
    if op not in _STRING_OPS:
        raise ValueError(
            f"Operator {op!r} is not supported for categorical column {col_name!r}. "
            f"Use '==' or '!='."
        )
    categories = col_stats["categories"]
    cmp = _STRING_OPS[op]
    valid = [c for c in categories if cmp(c, str(threshold))]
    if not valid:
        raise ValueError(
            f"No valid values for condition {col_name} {op} {threshold!r}. "
            f"Known categories: {categories}"
        )
    return valid


def _enumerate_numeric(
    col_name: str,
    op: str,
    threshold: float,
    col_stats: dict,
    float_precision: tp.Optional[int],
) -> tp.List[str]:
    """Enumerate valid numeric values by discretizing the column range."""
    if op not in _NUMERIC_OPS:
        raise ValueError(f"Unsupported operator {op!r} for numeric column {col_name!r}.")

    col_min = col_stats["min"]
    col_max = col_stats["max"]

    # Determine precision
    if float_precision is not None:
        prec = float_precision
    else:
        prec = _infer_precision(col_min, col_max)

    step = 10 ** (-prec) if prec > 0 else 1.0
    cmp = _NUMERIC_OPS[op]

    # Determine effective range boundaries
    eff_min = col_min
    eff_max = col_max

    # Estimate count and coarsen step if needed
    if step > 0:
        est_count = (eff_max - eff_min) / step + 1
        while est_count > _MAX_ENUM_VALUES and prec > 0:
            prec -= 1
            step = 10 ** (-prec) if prec > 0 else 1.0
            est_count = (eff_max - eff_min) / step + 1
        # If still too many after reaching precision 0, increase step
        if est_count > _MAX_ENUM_VALUES:
            step = (eff_max - eff_min) / (_MAX_ENUM_VALUES - 1)

    # Generate values
    values = []
    v = eff_min
    while v <= eff_max + step * 0.5:  # small tolerance for float rounding
        if cmp(v, threshold):
            values.append(_format_numeric(round(v, prec), float_precision))
        v += step

    if not values:
        raise ValueError(
            f"No valid values for condition {col_name} {op} {threshold}. "
            f"Column range is [{col_min}, {col_max}]."
        )

    # Deduplicate (formatting may collapse nearby values)
    seen = set()
    unique = []
    for val in values:
        if val not in seen:
            seen.add(val)
            unique.append(val)
    return unique


# ---------------------------------------------------------------------------
# Token Prefix Trie
# ---------------------------------------------------------------------------

class TrieNode:
    """A node in the token prefix trie."""
    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: tp.Dict[int, "TrieNode"] = {}
        self.is_terminal: bool = False


class TokenPrefixTrie:
    """Prefix trie over token ID sequences.

    Each path from root to a terminal node represents the complete token
    sequence for one valid value string.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: tp.List[int]):
        """Insert a token ID sequence into the trie."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_terminal = True

    def get_valid_next_tokens(self, prefix: tp.List[int]) -> tp.Tuple[tp.Set[int], bool]:
        """Get valid next token IDs given a prefix of already-generated tokens.

        Args:
            prefix: Token IDs generated so far for the current value.

        Returns:
            Tuple of (set of valid next token IDs, whether prefix is a complete value).
        """
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return set(), False
            node = node.children[tid]
        return set(node.children.keys()), node.is_terminal


def build_trie(valid_values: tp.List[str], tokenizer) -> TokenPrefixTrie:
    """Build a prefix trie from valid value strings.

    Each value is tokenized as ``" {value}"`` (with leading space) to match
    the generation context where the prompt ends with ``"{feature} is"``.

    Args:
        valid_values: List of valid value strings.
        tokenizer: HuggingFace tokenizer.

    Returns:
        TokenPrefixTrie containing all valid token sequences.
    """
    trie = TokenPrefixTrie()
    for val in valid_values:
        # Leading space matches generation: "feature is" + " value"
        token_ids = tokenizer.encode(f" {val}", add_special_tokens=False)
        if token_ids:
            trie.insert(token_ids)
    return trie


# ---------------------------------------------------------------------------
# LogitsProcessor for constrained decoding
# ---------------------------------------------------------------------------

class ConstrainedValueProcessor(LogitsProcessor):
    """LogitsProcessor that constrains generation to values in a prefix trie.

    Tracks the token IDs generated after the prompt and masks logits so only
    trie-valid continuations are allowed. When a complete value is formed,
    delimiter tokens (semicolon, comma, EOS) are also allowed.

    Args:
        trie: TokenPrefixTrie of valid value token sequences.
        tokenizer: HuggingFace tokenizer.
        prompt_length: Number of tokens in the input prompt (to know where
            the generated value starts).
    """

    def __init__(self, trie: TokenPrefixTrie, tokenizer, prompt_length: int):
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

        # Pre-compute delimiter token IDs
        self._delimiter_ids: tp.Set[int] = set()
        for delim in [";", ",", "\n"]:
            ids = tokenizer.encode(delim, add_special_tokens=False)
            if ids:
                self._delimiter_ids.add(ids[0])
            # Also try with leading space
            ids_space = tokenizer.encode(f" {delim}", add_special_tokens=False)
            if ids_space:
                self._delimiter_ids.add(ids_space[0])
        # Add EOS token
        if tokenizer.eos_token_id is not None:
            self._delimiter_ids.add(tokenizer.eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Mask logits to only allow trie-valid continuations.

        Args:
            input_ids: Shape ``(batch_size, seq_len)`` — current token sequence.
            scores: Shape ``(batch_size, vocab_size)`` — logits for the next token.

        Returns:
            Modified scores with invalid tokens set to ``-inf``.
        """
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # Tokens generated so far (after prompt)
            generated = input_ids[i, self.prompt_length:].tolist()

            # If a delimiter was already emitted, the value is done —
            # stop constraining and let the model generate freely.
            if any(t in self._delimiter_ids for t in generated):
                continue

            valid_next, is_complete = self.trie.get_valid_next_tokens(generated)

            # Build set of allowed tokens
            allowed = set(valid_next)
            if is_complete:
                allowed |= self._delimiter_ids

            if not allowed:
                # Safety fallback: allow delimiters so generation can terminate
                logging.warning(
                    "Constrained trie has no valid continuation; allowing delimiters as fallback."
                )
                allowed = self._delimiter_ids

            if allowed:
                mask = torch.full_like(scores[i], float("-inf"))
                for tid in allowed:
                    if 0 <= tid < scores.shape[1]:
                        mask[tid] = 0.0
                scores[i] = scores[i] + mask

        return scores
