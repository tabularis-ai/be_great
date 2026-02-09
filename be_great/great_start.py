import random
import numpy as np
import typing as tp


def _pad(x, length: int, pad_value: int):
    """
    Prepend the pad value until the array reaches the specific length
    """
    return [pad_value] * (length - len(x)) + x


def _pad_tokens(tokens, pad_value: int):
    """
    Checks that all tensors in the list have the same length, pads them if necessary to the max length

    Args:
        tokens: List of Tensors
        pad_value: Token ID to use for padding

    Returns:
        List of Tensors, where each Tensor has the same length
    """
    max_length = len(max(tokens, key=len))
    tokens = [_pad(t, max_length, pad_value) for t in tokens]
    return tokens


class GReaTStart:
    """Abstract super class GReaT Start

    GReaT Start creates tokens to start the generation process.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
    """

    def __init__(self, tokenizer):
        """
        Initializes the super class.

        Args:
            tokenizer: Tokenizer from the HuggingFace library
        """
        self.tokenizer = tokenizer

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        """Get Start Tokens

        Creates starting points for the generation process

        Args:
            n_samples: Number of start prompts to create

        Returns:
            List of n_sample lists with tokens
        """
        raise NotImplementedError("This has to be overwritten but the subclasses")


class CategoricalStart(GReaTStart):
    """Categorical Starting Feature

    A categorical column with its categories is used as starting point.

    Attributes:
        start_col (str): Name of the categorical column
        population (list[str]): Possible values the column can take
        weights (list[float]): Probabilities for the individual categories

    """

    def __init__(self, tokenizer, start_col: str, start_col_dist: dict):
        """Initializes the Categorical Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the categorical column
            start_col_dist: Distribution of the categorical column (dict of form {"Cat A": 0.8, "Cat B": 0.2})
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, dict), ""

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        start_text = [self.start_col + " is " + str(s) + "," for s in start_words]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"], pad_value=pad_id)
        return start_tokens


class ContinuousStart(GReaTStart):
    """Continuous Starting Feature

    A continuous column with some noise is used as starting point.

    Attributes:
        start_col (str): Name of the continuous column
        start_col_dist (list[float]): The continuous column from the train data set
        noise (float): Size of noise that is added to each value
        decimal_places (int): Number of decimal places the continuous values have
    """

    def __init__(
        self,
        tokenizer,
        start_col: str,
        start_col_dist: tp.List[float],
        noise: float = 0.01,
        decimal_places: int = 5,
    ):
        """Initializes the Continuous Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the continuous column
            start_col_dist: The continuous column from the train data set
            noise: Size of noise that is added to each value
            decimal_places: Number of decimal places the continuous values have
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, list), ""

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        # start_words += np.random.normal(size=n_samples) * self.noise  # add noise to start words
        start_text = [
            self.start_col + " is " + format(s, f".{self.decimal_places}f") + ","
            for s in start_words
        ]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"], pad_value=pad_id)
        return start_tokens


class RandomStart(GReaTStart):
    """Random Starting Features

    Random column names are used as start point. Can be used if no distribution of any column is known.

    Attributes:
        all_columns (List[str]): Names of all columns
    """

    def __init__(self, tokenizer, all_columns: tp.List[str]):
        """Initializes the Random Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            all_columns: Names of all columns
        """
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_text = [s + " is " for s in start_words]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        start_tokens = _pad_tokens(self.tokenizer(start_text)["input_ids"], pad_value=pad_id)
        return start_tokens
