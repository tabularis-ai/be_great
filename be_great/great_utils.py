import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(
    data: tp.Union[pd.DataFrame, np.ndarray], columns=None
) -> pd.DataFrame:
    """Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(
        data, np.ndarray
    ), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert (
        columns
    ), "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(
        data[0]
    ), "%d column names are given, but array has %d columns!" % (
        len(columns),
        len(data[0]),
    )

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_text(
    tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer
) -> tp.List[str]:
    """Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data(
    text: tp.List[str], columns: tp.List[str]
) -> pd.DataFrame:
    """Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        columns: Column names of the data

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    generated = []

    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns, "placeholder")

        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" is ")
            if values[0] in columns and td[values[0]] == "placeholder":
                try:
                    td[values[0]] = values[1]
                except IndexError:
                    # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
        generated.append(td)
    df_gen = pd.DataFrame(generated)
    df_gen.replace("None", None, inplace=True)

    return df_gen


def _encode_row_partial(row, shuffle=True, float_precision=None):
    """Function that takes a row and converts all columns into the text representation that are not NaN.
    
    Args:
        row: Pandas Series row
        shuffle: Whether to shuffle the order of columns
        float_precision: Number of decimal places to use for floating point numbers. If None, full precision is used.
    
    Returns:
        String representation of the row with 'column is value' format
    """
    num_cols = len(row.index)
    if not shuffle:
        idx_list = np.arange(num_cols)
    else:
        idx_list = np.random.permutation(num_cols)

    parts = []
    for i in idx_list:
        value = row[row.index[i]]
        if not pd.isna(value):
            col_name = row.index[i]
            if isinstance(value, (float, np.floating)) and float_precision is not None:
                # Format to a string with specified decimal places, removing trailing zeros
                formatted_value_str = f"{value:.{float_precision}f}"
                if '.' in formatted_value_str:
                    formatted_value_str = formatted_value_str.rstrip('0').rstrip('.')
                final_value_representation = formatted_value_str
            else:
                final_value_representation = value
            parts.append(f"{col_name} is {final_value_representation}")

    return ", ".join(parts)


def _get_random_missing(row):
    """Return a random missing column or None if all columns are filled."""
    nans = list(row[pd.isna(row)].index)
    return np.random.choice(nans) if len(nans) > 0 else None


def _partial_df_to_promts(partial_df: pd.DataFrame, float_precision=None):
    """Convert DataFrame with missingvalues to a list of starting promts for GReaT
        Args:
        partial_df: Pandas DataFrame to be imputed where missing values are encoded by NaN.
        float_precision: Number of decimal places to use for floating point numbers. If None, full precision is used.

    Returns:
        List of strings with the starting prompt for each sample.
    """
    encoder = lambda x: _encode_row_partial(x, True, float_precision)
    res_encode = list(partial_df.apply(encoder, axis=1))
    res_first = list(partial_df.apply(_get_random_missing, axis=1))

    # Edge case: all values are missing, will return empty string which is not supported.
    # Use first attribute as starting prompt.
    # default_promt = partial_df.columns[0] + " is "
    res = [
        ((enc + ", ") if len(enc) > 0 else "")
        + (fst + " is" if fst is not None else "")
        for enc, fst in zip(res_encode, res_first)
    ]
    return res


class bcolors:
    """
    We love colors, you?
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
