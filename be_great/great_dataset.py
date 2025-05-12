import random
import typing as tp
import numpy as np

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
        float_precision (int, optional): Number of decimal places to use for floating point numbers.
                                        If None, full precision is used.
    """

    def set_tokenizer(self, tokenizer, float_precision=None):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
            float_precision: Number of decimal places to use for floating point numbers.
                           If None, full precision is used.
        """
        self.tokenizer = tokenizer
        self.float_precision = float_precision

    def _format_value(self, value):
        """Format a value based on its type.
        
        For floats, applies precision formatting if float_precision is set.
        
        Args:
            value: The value to format
            
        Returns:
            Formatted string value
        """
        if isinstance(value, (float, np.floating)) and self.float_precision is not None:
            # Format to a string with specified decimal places, removing trailing zeros
            formatted_value_str = f"{value:.{self.float_precision}f}"
            if '.' in formatted_value_str:
                formatted_value_str = formatted_value_str.rstrip('0').rstrip('.')
            return formatted_value_str
        return str(value).strip()

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        shuffled_text = ", ".join(
            [
                "%s is %s"
                % (row.column_names[i], self._format_value(row.columns[i].to_pylist()[0]))
                for i in shuffle_idx
            ]
        )
        tokenized_text = self.tokenizer(shuffled_text, padding=True)
        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
