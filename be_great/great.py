from typing import List
import warnings
import json
import typing as tp
import logging

import fsspec
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    _pad_tokens,
)
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_promts,
    bcolors,
)


class GReaT:
    """GReaT Class

    The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        batch_size: int = 8,
        target_modules : List[str] = None,
        efficient_finetuning: str = "",
        **train_kwargs,
    ):
        """Initializes GReaT.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            efficient_finetuning: Indication of fune-tuning method
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        if self.efficient_finetuning == "lora":
            # Lazy importing
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_int8_training,
                    TaskType,
                )
            except ImportError:
                raise ImportError(
                    "This function requires the 'peft' package. Please install it with - pip install peft==0.9.0"
                )

            if not target_modules:
                target_modules = []                
                for name, module in self.model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d)):
                        target_modules.append('.'.join(name.split('.')[4:]).split('.')[0])
                
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,  # only training 0.16% of the parameters of the model
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_int8_training(self.model)
            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> GReaTTrainer:
        """Fine-tune GReaT using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
        )

        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Generate synthetic tabular data samples.

        Args:
            n_samples (int): Number of synthetic samples to generate.
            start_col (str, optional): Feature to use as the starting point for the generation process.
                Defaults to the target learned during fitting if not provided.
            start_col_dist (dict or list, optional): Feature distribution of the starting feature.
                For discrete columns, should be in the format "{F1: p1, F2: p2, ...}".
                For continuous columns, should be a list of possible values.
                Defaults to the target distribution learned during fitting if not provided.
            temperature (float): Controls the softmax function for token sampling.
                Lower values make it sharper (0 equals greedy search), higher values introduce more diversity but also uncertainty.
            k (int): Sampling batch size. Higher values speed up the generation process.
            max_length (int): Maximum number of tokens to generate. Ensure it's long enough to not cut off any information.
            drop_nan (bool): Whether to drop rows with NaN values. Defaults to False.
            device (str): Device to use for generation. Set to "cpu" to avoid using GPU. Specific GPU can also be named.

        Returns:
            pd.DataFrame: DataFrame containing n_samples rows of generated data.
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self.model.to(device)

        # Init list for generated DataFrames
        dfs = []

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            _cnt = 0
            try:
                while n_samples > already_generated:
                    start_tokens = great_start.get_start_tokens(k)
                    start_tokens = torch.tensor(start_tokens).to(device)

                    # Generate tokens
                    tokens = self.model.generate(
                        input_ids=start_tokens,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=50256,
                    )

                    # Convert tokens back to tabular data
                    text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(text_data, self.columns)

                    # Remove rows where we have not generated anything
                    df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]

                    # Remove rows where all values are NaN
                    df_gen = df_gen.dropna(how="all")

                    # Optional: Remove rows with any NaN values
                    if drop_nan:
                        df_gen = df_gen.dropna()

                    # Remove rows with flawed numerical values but keep NaNs
                    for i_num_cols in self.num_cols:
                        coerced_series = pd.to_numeric(
                            df_gen[i_num_cols], errors="coerce"
                        )
                        df_gen = df_gen[
                            coerced_series.notnull() | df_gen[i_num_cols].isna()
                        ]

                    # Convert numerical columns to float
                    df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                    dfs.append(df_gen)
                    already_generated += len(dfs[-1])

                    # Update progress bar
                    pbar.update(len(dfs[-1]))

                    # Check if we are actually generating synthetic samples and if not, break everything
                    _cnt += 1
                    if _cnt > 13 and already_generated == 0:
                        raise Exception("Breaking the generation loop!")

            except Exception as e:
                print(f"{bcolors.FAIL}An error has occurred: {str(e)}{bcolors.ENDC}")
                print(
                    f"{bcolors.WARNING}To address this issue, consider fine-tuning the GReaT model for an longer period. This can be achieved by increasing the number of epochs.{bcolors.ENDC}"
                )
                print(
                    f"{bcolors.WARNING}Alternatively, you might consider increasing the max_length parameter within the sample function. For example: model.sample(n_samples=10, max_length=2000){bcolors.ENDC}"
                )
                print(
                    f"{bcolors.OKBLUE}If the problem persists despite these adjustments, feel free to raise an issue on our GitHub page at: https://github.com/kathrinse/be_great/issues{bcolors.ENDC}"
                )

        df_gen = pd.concat(dfs)
        df_gen = df_gen.reset_index(drop=True)
        return df_gen.head(n_samples)

    def great_sample(
        self,
        starting_prompts: tp.Union[str, list[str]],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        self.model.to(device)
        starting_prompts = (
            [starting_prompts]
            if isinstance(starting_prompts, str)
            else starting_prompts
        )
        generated_data = []

        # Generate a sample for each starting point
        if len(starting_prompts) > 1:
            loop_iter = tqdm(starting_prompts)
        else:
            loop_iter = starting_prompts
        for prompt in loop_iter:
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # Generate tokens
            gen = self.model.generate(
                input_ids=torch.unsqueeze(start_token, 0),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=50256,
            )
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, self.columns)

        return df_gen

    def impute(
        self,
        df_miss: pd.DataFrame,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        max_retries=15,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """Impute a DataFrame with missing values using a trained GReaT model.
        Args:
            df_miss: pandas data frame of the exact same format (column names, value ranges/types) as the data that
             was used to train the GReaT model, however some values might be missing, which is indicated by the value of NaN.
             This function will sample the missing values conditioned on the remaining values.
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the specific GPU to run on.

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """

        # Check DataFrame passed.
        if set(df_miss.columns) != set(self.columns):
            raise ValueError(
                "The column names in the DataFrame passed to impute do not match the columns of the GReaT model."
            )

        self.model.to(device)

        # start_token = torch.tensor(_pad_tokens(self.tokenizer(starting_prompts)["input_ids"])).to(device)
        index = 0
        df_list = []
        with tqdm(total=len(df_miss)) as pbar:
            while index < len(df_miss):
                is_complete = False
                retries = 0
                df_curr = df_miss.iloc[[index]]
                org_index = df_curr.index  # Keep index in new DataFrame
                while not is_complete:
                    num_attrs_missing = pd.isna(df_curr).sum().sum()
                    # print("Number of missing values: ",  num_attrs_missing)
                    # Generate text promt from current features.
                    starting_prompts = _partial_df_to_promts(df_curr)
                    df_curr = self.great_sample(
                        starting_prompts, temperature, max_length, device=device
                    )

                    # Convert numerical values to float, flawed numerical values to NaN
                    for i_num_cols in self.num_cols:
                        df_curr[i_num_cols] = pd.to_numeric(
                            df_curr[i_num_cols], errors="coerce"
                        )
                    df_curr[self.num_cols] = df_curr[self.num_cols].astype(np.float)

                    # Check for missing values
                    nans = df_curr.isna()
                    if not df_curr.isna().any().any():
                        is_complete = True
                        df_list.append(df_curr.set_index(org_index))
                    else:
                        retries += 1
                    if retries == max_retries:
                        warnings.warn("Max retries reached.")
                        break
                index += 1
                pbar.update(1)
        return pd.concat(df_list, axis=0)

    def save(self, path: str):
        """Save GReaT Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        if fs.exists(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            fs.mkdir(path)

        # Save attributes
        with fs.open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(
                    attributes["conditional_col_dist"]
                )

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), fs.open(path + "/model.pt", "wb"))

    def load_finetuned_model(self, path: str):
        """Load fine-tuned model

        Load the weights of a fine-tuned large language model into the GReaT pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(fsspec.open(path, "rb")))

    @classmethod
    def load_from_dir(cls, path: str):
        """Load GReaT class

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        assert fs.exists(path), f"Directory {path} does not exist."

        # Load attributes
        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(fs.open(path + "/model.pt", "rb"), map_location="cpu"))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(
        self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None
    ):
        assert conditional_col is None or isinstance(
            conditional_col, str
        ), f"The column name has to be a string and not {type(conditional_col)}"
        assert (
            conditional_col is None or conditional_col in df.columns
        ), f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(
        self,
        start_col: tp.Optional[str],
        start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]],
    ) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(
                f"Start column {start_col} was given, but no corresponding distribution."
            )
        if start_col_dist is not None and not start_col:
            raise ValueError(
                f"Start column distribution {start_col} was given, the column name is missing."
            )

        assert start_col is None or isinstance(
            start_col, str
        ), f"The column name has to be a string and not {type(start_col)}"
        assert (
            start_col_dist is None
            or isinstance(start_col_dist, dict)
            or isinstance(start_col_dist, list)
        ), f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)
