import warnings
import json
import typing as tp
import logging
import re
import random

import fsspec
import numpy as np
import pandas as pd

from chugchug.compat import tqdm
from chugchug import Chug

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, LogitsProcessorList

from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    _pad_tokens,
)
from be_great.great_trainer import GReaTTrainer
from be_great.great_constrained import (
    parse_condition,
    enumerate_valid_values,
    build_trie,
    ConstrainedValueProcessor,
    compute_value_weights,
    first_token_log_bias,
)
from be_great.great_mock_datasets import (
    populate_schema_state,
    build_effective_conditions,
    apply_null_probabilities,
    build_few_shot_prefix,
    set_global_seed,
)
from be_great.great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_prompts,
    bcolors,
)


class ChugProgressCallback(TrainerCallback):
    """🚂 chugchug-backed progress bar for HuggingFace Trainer.

    Replaces HF's default tqdm `ProgressCallback` with a gradient bar that
    auto-colors metrics (loss green when improving, red when worsening).
    """

    def __init__(self, gradient: str = "fire", desc: str = "Training"):
        self.gradient = gradient
        self.desc = desc
        self.bar: tp.Optional[Chug] = None
        self._last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        self.bar = Chug(total=total, desc=self.desc, gradient=self.gradient, unit="step")
        self._last_step = 0
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is None:
            return control
        delta = state.global_step - self._last_step
        if delta > 0:
            self.bar.update(delta)
            self._last_step = state.global_step
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.bar is None or not logs:
            return control
        metrics = {}
        if "loss" in logs:
            metrics["loss"] = f"{logs['loss']:.4f}"
        if "learning_rate" in logs:
            metrics["lr"] = f"{logs['learning_rate']:.2e}"
        if "epoch" in logs:
            metrics["epoch"] = f"{logs['epoch']:.2f}"
        if metrics:
            self.bar.set_metrics(**metrics)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        return control


class LiveQualityCallback(TrainerCallback):
    """Live synthesis-quality eval during training.

    Periodically samples from the (in-training) model, compares the sample
    against a held-out evaluation set using ``ColumnShapes`` similarity
    (1.0 = identical distributions, 0.0 = totally different), and pushes
    the score into the chugchug progress bar as ``quality=…``.

    Args:
        great_model: The owning ``GReaT`` instance (needed for ``.sample()``).
        eval_data: Held-out ``pd.DataFrame`` used as the comparison target.
        chug_callback: The ``ChugProgressCallback`` whose bar should receive
            the live ``quality`` metric.
        eval_every: If ``None`` (default), evaluate once at the end of each
            epoch. If an int, evaluate every N training steps.
        eval_n_samples: Number of synthetic rows to draw per evaluation.
            Larger = more stable score, slower. Default 50.
        max_length: Max generation length per row. Default 100.
    """

    def __init__(
        self,
        great_model,
        eval_data: pd.DataFrame,
        chug_callback: ChugProgressCallback,
        eval_every: tp.Optional[int] = None,
        eval_n_samples: int = 50,
        max_length: int = 100,
    ):
        from be_great.metrics import ColumnShapes  # lazy: avoid import cycles
        self.great = great_model
        self.eval_data = eval_data
        self.chug_cb = chug_callback
        self.eval_every = eval_every
        self.eval_n_samples = eval_n_samples
        self.max_length = max_length
        self._metric = ColumnShapes()

    def _evaluate(self, step: int):
        was_training = self.great.model.training
        try:
            self.great.model.eval()
            current_device = next(self.great.model.parameters()).device
            # Sync GReaT.device with reality so sample() does not move the model
            self.great.device = current_device
            with torch.no_grad():
                synth = self.great.sample(
                    n_samples=self.eval_n_samples,
                    max_length=self.max_length,
                    k=min(self.eval_n_samples, 20),
                    device=str(current_device),
                )
            result = self._metric.compute(self.eval_data, synth)
            score = float(result.get("column_shapes_mean", 0.0))
            if self.chug_cb is not None and self.chug_cb.bar is not None:
                # Short key to fit narrow terminals (q = column_shapes_mean)
                self.chug_cb.bar.set_metrics(q=f"{score:.3f}")
            print(f"  🎯 [quality @ step {step}]  column_shapes_mean = {score:.4f}", flush=True)
        except Exception as e:
            logging.warning(f"LiveQuality eval failed at step {step}: {e}")
        finally:
            if was_training:
                self.great.model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_every is None:
            return control
        if state.global_step > 0 and state.global_step % self.eval_every == 0:
            self._evaluate(state.global_step)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_every is None:
            self._evaluate(state.global_step)
        return control


class RandomConditionalColumnCallback(TrainerCallback):
    """Callback to randomly change the conditional column after each epoch."""
    
    def __init__(self, model, dataframe):
        self.model = model
        self.dataframe = dataframe
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Randomly change the conditional column at the end of each epoch."""
        # Choose a random column and update conditional information
        random_column = random.choice(self.model.columns)
        self.model._update_conditional_information(self.dataframe, random_column)
        logging.info(f"Changed conditional column to: {random_column}")
        return control


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
        efficient_finetuning: str = "",
        lora_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
        float_precision: tp.Optional[int] = None,
        report_to: tp.List[str] = [],
        **train_kwargs,
    ):
        """Initializes GReaT.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            efficient_finetuning: Indication of fine-tuning method. Set to "lora" to enable LoRA fine-tuning.
            lora_config: Optional dictionary of LoRA hyperparameters to override defaults.
                Supported keys:
                - r (int): LoRA rank. Default 16.
                - lora_alpha (int): LoRA alpha scaling factor. Default 32.
                - target_modules (list[str] | None): List of module names to apply LoRA to.
                    If None, auto-detected based on the model architecture.
                - lora_dropout (float): Dropout probability for LoRA layers. Default 0.05.
                - bias (str): Bias type for LoRA. Default "none".
                - task_type (str): PEFT task type. Default "CAUSAL_LM".
                - modules_to_save (list[str] | None): Additional modules to save alongside LoRA. Default None.
            float_precision: Number of decimal places to use for floating point numbers. If None, full precision is used.
            report_to: List of integrations to report to. Empty list means no reporting (disable Weights & Biases).
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        # Only fall back to eos_token if the tokenizer has no pad_token at all
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models need left-padding for correct generation
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Store the user-provided LoRA config dict (used for serialization)
        self._lora_config_dict = lora_config or {}

        if self.efficient_finetuning == "lora":
            self._apply_lora(self._lora_config_dict)

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = {"report_to": report_to, **train_kwargs}

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

        # Device management — resolved once, reused everywhere
        self.device = None

        # Store float precision setting
        self.float_precision = float_precision

        # Per-column statistics for constrained sampling
        self.col_stats = None

    def __repr__(self) -> str:
        fitted = self.columns is not None
        if fitted:
            cols_info = (
                f"fitted on {len(self.columns)} cols "
                f"({len(self.num_cols)} num, {len(self.columns) - len(self.num_cols)} cat)"
            )
        else:
            cols_info = "unfitted"
        device = str(self.device) if self.device is not None else "unresolved"
        peft = ", peft=lora" if self.efficient_finetuning == "lora" else ""
        return (
            f"GReaT(llm={self.llm!r}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, {cols_info}, device={device}{peft})"
        )

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_target_modules(model) -> tp.List[str]:
        """Auto-detect linear attention projection modules for LoRA.

        Inspects the model's named modules and returns a list of common
        attention-projection module name patterns found across popular
        architectures (GPT-2, LLaMA/Mistral, Falcon, GPT-NeoX, Bloom, …).

        Returns:
            List of module name strings suitable for ``target_modules`` in
            a ``LoraConfig``.
        """
        # Candidate module-name patterns grouped by architecture family.
        # Order matters: we check from most common to least common.
        candidate_patterns = [
            # LLaMA / Mistral / Mixtral / Gemma / Phi-3 style
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            # GPT-2 style (fused QKV)
            ["c_attn", "c_proj"],
            ["c_attn"],
            # Falcon style
            ["query_key_value", "dense"],
            ["query_key_value"],
            # GPT-NeoX / Pythia style
            ["query_key_value", "dense"],
            # Bloom style
            ["query_key_value", "dense"],
            # GPT-J style
            ["q_proj", "k_proj", "v_proj", "out_proj"],
            # OPT style
            ["q_proj", "v_proj", "k_proj", "out_proj"],
        ]

        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

        for pattern in candidate_patterns:
            if all(p in module_names for p in pattern):
                logging.info(f"Auto-detected LoRA target modules: {pattern}")
                return pattern

        # Fallback: find all nn.Linear leaf modules that look like
        # attention projections (name contains "attn", "attention", or "proj").
        import torch.nn as nn

        fallback = set()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                short = name.split(".")[-1]
                if any(kw in short.lower() for kw in ("attn", "attention", "proj", "query", "key", "value")):
                    fallback.add(short)
        if fallback:
            result = sorted(fallback)
            logging.info(f"Auto-detected LoRA target modules (fallback): {result}")
            return result

        raise ValueError(
            "Could not auto-detect target modules for LoRA. "
            "Please specify them explicitly via lora_config={'target_modules': ['module_name', ...]}."
        )

    def _apply_lora(self, lora_cfg: tp.Dict[str, tp.Any]):
        """Wrap ``self.model`` with a PEFT LoRA adapter.

        Args:
            lora_cfg: Dictionary of LoRA hyperparameters (may be empty to
                use all defaults).
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "LoRA fine-tuning requires the 'peft' package. "
                "Install it with:  pip install peft"
            )

        # Resolve target modules: user-supplied or auto-detected
        target_modules = lora_cfg.get("target_modules", None)
        if target_modules is None:
            target_modules = self._detect_target_modules(self.model)

        # Map string task type to enum if needed
        task_type_raw = lora_cfg.get("task_type", "CAUSAL_LM")
        if isinstance(task_type_raw, str):
            task_type = getattr(TaskType, task_type_raw, TaskType.CAUSAL_LM)
        else:
            task_type = task_type_raw

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=task_type,
            modules_to_save=lora_cfg.get("modules_to_save", None),
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    @property
    def _is_peft_model(self) -> bool:
        """Return True if the current model is wrapped with PEFT."""
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve and cache the compute device, moving the model if needed.

        Args:
            device: Requested device string. Use ``"auto"`` to pick the best
                available backend (CUDA > MPS > CPU). Other accepted values:
                ``"cuda"``, ``"cuda:N"``, ``"mps"``, ``"cpu"``. If a requested
                accelerator is unavailable, falls back to CPU with a warning.

        Returns:
            torch.device that the model is now on.
        """
        cuda_ok = torch.cuda.is_available()
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        if device == "auto":
            if cuda_ok:
                resolved = torch.device("cuda")
            elif mps_ok:
                resolved = torch.device("mps")
            else:
                resolved = torch.device("cpu")
            logging.info(f"Auto-detected device: {resolved}")
        elif device.startswith("cuda"):
            resolved = torch.device(device if cuda_ok else "cpu")
            if not cuda_ok:
                logging.warning(f"CUDA requested but not available — falling back to CPU.")
        elif device == "mps":
            resolved = torch.device("mps" if mps_ok else "cpu")
            if not mps_ok:
                logging.warning("MPS requested but not available — falling back to CPU.")
        else:
            resolved = torch.device(device)

        if self.device != resolved:
            self.device = resolved
            self.model.to(self.device)
            logging.info(f"Model moved to {self.device}")
        return self.device

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
        random_conditional_col: bool = True,
        eval_data: tp.Optional[pd.DataFrame] = None,
        eval_every: tp.Optional[int] = None,
        eval_n_samples: int = 50,
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
            random_conditional_col: If True, a different random column will be selected for preconditioning
            in each epoch. This helps prevent any single column from being overfitted during training.
            eval_data: Optional held-out DataFrame for live synthesis-quality eval. When provided, a
                ``column_shapes_mean`` similarity score is computed periodically and shown live in the
                chugchug progress bar as ``quality=…`` (1.0 = real vs synthetic indistinguishable).
            eval_every: If ``None`` (default), live eval runs at the end of every epoch. If an int,
                live eval runs every N training steps. Ignored if ``eval_data is None``.
            eval_n_samples: Number of synthetic rows to draw per live-eval call. Default 50.

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        # Disable HF's default tqdm so chugchug owns the progress bar
        # (user can override by passing disable_tqdm=False in train_kwargs)
        self.train_hyperparameters.setdefault("disable_tqdm", True)
        # pin_memory only helps CUDA; disable on MPS/CPU to silence the warning
        if not torch.cuda.is_available():
            self.train_hyperparameters.setdefault("dataloader_pin_memory", False)

        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )

        # Set up callbacks
        chug_cb = ChugProgressCallback()
        callbacks = [chug_cb]
        if random_conditional_col:
            logging.info("Random conditional column enabled. Will randomly select a different column for each epoch.")
            callbacks.append(RandomConditionalColumnCallback(self, df))
        if eval_data is not None:
            logging.info(
                f"Live quality eval enabled (n_samples={eval_n_samples}, "
                f"every={'epoch' if eval_every is None else f'{eval_every} steps'})."
            )
            callbacks.append(LiveQualityCallback(
                great_model=self,
                eval_data=eval_data,
                chug_callback=chug_cb,
                eval_every=eval_every,
                eval_n_samples=eval_n_samples,
            ))
        
        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            processing_class=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
            callbacks=callbacks,
        )

        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def mock(
        self,
        schema: tp.Dict[str, tp.Dict[str, tp.Any]],
        n_samples: int = 10,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
        examples: tp.Optional[tp.Sequence[tp.Mapping[str, tp.Any]]] = None,
        seed: tp.Optional[int] = None,
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "auto",
        random_feature_order: bool = True,
        conditional_col: tp.Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate mock tabular data **without fitting on real data**.

        Provide a schema describing the columns; GReaT will populate the internal
        state that ``fit()`` normally builds, then run guided/constrained sampling
        through the existing pipeline. Useful for privacy-safe dummy data, test
        fixtures, and schema prototyping. Works best when ``llm`` is a model already
        pre-trained on tabular GReaT-format rows (e.g. ``tabularisai/Qwen3-0.3B-distil``).

        Args:
            schema: Dict mapping ``column_name -> spec_dict``. Specs:
                - Numerical: ``{"type": "num", "range": (min, max)}``
                - Categorical: ``{"type": "cat", "values": ["A", "B", ...]}``
                Optional per-column: ``integer``, ``precision``, ``dist`` +
                ``mean`` + ``std`` (numeric), ``weights`` (categorical),
                ``null_prob`` (both).
            n_samples: Number of mock rows to generate.
            conditions: Optional ``{col: condition_string}``, e.g.
                ``{"age": ">= 40", "sex": "!= 'Male'"}``. Same syntax as ``sample()``.
            examples: Optional list of dict rows used as few-shot context.
                Each row is formatted as a GReaT-style example and prepended to
                the generation prompt. Improves realism for non-tabular-pretrained
                LLMs (e.g. vanilla GPT-2). Two to five rows is usually plenty.
            seed: If set, seeds ``random``, ``numpy``, and ``torch`` so the
                generated DataFrame is reproducible across runs.
            temperature: Generation temperature.
            max_length: Max tokens per row.
            device: ``"auto"`` (default) picks CUDA > MPS > CPU.
            random_feature_order: Randomize feature order per row when sampling.
            conditional_col: Which column to use as the generation starting point.
                Defaults to the last column in the schema.

        Returns:
            pd.DataFrame with ``n_samples`` rows matching the schema. NaNs
            appear in any column whose schema declared ``null_prob > 0``.
        """
        rng = set_global_seed(seed) if seed is not None else np.random.default_rng()

        populate_schema_state(self, schema, conditional_col=conditional_col)
        effective_conditions = build_effective_conditions(self.col_stats, conditions)
        few_shot_prefix = build_few_shot_prefix(examples or [], self.columns)
        if few_shot_prefix:
            logging.info(
                f"Using {len(examples)} few-shot example row(s) as prompt prefix."
            )

        df = self._guided_sample(
            n_samples=n_samples,
            temperature=temperature,
            max_length=max_length,
            device=device,
            random_feature_order=random_feature_order,
            conditions=effective_conditions,
            examples_prefix=few_shot_prefix,
        )
        return apply_null_probabilities(df, self.col_stats, rng=rng)

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "auto",
        guided_sampling: bool = False,
        random_feature_order: bool = True,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
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
            device (str): Device to use for generation. Defaults to "auto" (picks cuda > mps > cpu).
                Set to "cpu", "cuda", "cuda:N", or "mps" to force a specific device.
            guided_sampling (bool): Whether to use guided feature-by-feature sampling (True) or the legacy approach (False).
                Note that guided sampling may be slower but can be more reliable for certain datasets.
            random_feature_order (bool): Whether to randomize feature order for each sample in guided sampling.
            conditions (dict, optional): Dictionary mapping column names to condition strings.
                For example: ``{"Age": ">= 30", "City": "!= 'New York'"}``.
                When provided, guided sampling is automatically enabled.

        Returns:
            pd.DataFrame: DataFrame containing n_samples rows of generated data.
        """
        # Validate and handle conditions
        if conditions:
            # Validate column names
            if self.columns is None:
                raise ValueError("Model has not been fitted yet. Please call fit() first.")
            for col in conditions:
                if col not in self.columns:
                    raise ValueError(
                        f"Condition column {col!r} not found in model columns: {self.columns}"
                    )
            # Validate col_stats availability
            if self.col_stats is None:
                raise ValueError(
                    "Column statistics (col_stats) are not available. "
                    "Please re-fit the model so that column statistics are computed."
                )
            # Auto-enable guided sampling
            if not guided_sampling:
                logging.info(
                    "Conditions provided; automatically enabling guided_sampling=True."
                )
                guided_sampling = True

        # Choose the sampling method
        if guided_sampling:
            return self._guided_sample(
                n_samples=n_samples,
                temperature=temperature,
                max_length=max_length,
                device=device,
                random_feature_order=random_feature_order,
                conditions=conditions,
            )
        else:
            return self._legacy_sample(
                n_samples=n_samples,
                start_col=start_col,
                start_col_dist=start_col_dist,
                temperature=temperature,
                k=k,
                max_length=max_length,
                drop_nan=drop_nan,
                device=device,
            )

    def _guided_sample(
        self,
        n_samples: int = 10,
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "auto",
        random_feature_order: bool = True,
        conditions: tp.Optional[tp.Dict[str, str]] = None,
        examples_prefix: str = "",
    ) -> pd.DataFrame:
        """
        Generate synthetic data with guided feature name prompting.

        Args:
            n_samples (int): Number of samples to generate
            temperature (float): Temperature for sampling
            max_length (int): Maximum length of generated tokens
            device (str): Device to use for generation. Defaults to "auto" (picks cuda > mps > cpu).
            random_feature_order (bool): Whether to randomize feature order for each sample
            conditions (dict, optional): Dictionary mapping column names to condition strings.

        Returns:
            pd.DataFrame: Synthetic data with original column names
        """
        if self.columns is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")

        # Extract known categorical values - if we can find original distributions
        categorical_values = {}
        cat_cols = [col for col in self.columns if col.startswith('cat_')]

        # Try to infer categorical values from distributions
        for col in cat_cols:
            if col == self.conditional_col and isinstance(self.conditional_col_dist, dict):
                categorical_values[col] = list(self.conditional_col_dist.keys())

        # Make sure we're in eval mode and use the specified device
        self._resolve_device(device)
        self.model.eval()

        # Pre-build tries for constrained columns (cached, not rebuilt per sample)
        constraint_tries = {}
        first_token_biases: tp.Dict[str, tp.Dict[int, float]] = {}
        if conditions:
            for col, cond_str in conditions.items():
                op, threshold = parse_condition(cond_str)
                valid_values = enumerate_valid_values(
                    col, op, threshold, self.col_stats[col], self.float_precision
                )
                constraint_tries[col] = build_trie(valid_values, self.tokenizer)
                logging.info(
                    f"Built constraint trie for {col!r} ({cond_str}): "
                    f"{len(valid_values)} valid values"
                )
                # Compute first-token bias from schema dist/weights (if any)
                v_weights = compute_value_weights(self.col_stats[col], valid_values)
                if v_weights:
                    first_token_biases[col] = first_token_log_bias(
                        valid_values, v_weights, self.tokenizer
                    )
                    logging.info(
                        f"Applied first-token bias for {col!r} from schema "
                        f"({self.col_stats[col].get('dist') or 'weights'})"
                    )

        synthetic_data = []

        # Use tqdm for progress tracking
        with tqdm(total=n_samples) as pbar:
            for i in range(n_samples):
                try:
                    # Get feature names
                    feature_names = self.columns.copy()

                    # Randomize feature order if requested
                    if random_feature_order:
                        random.shuffle(feature_names)

                    # Few-shot example rows prepended as in-context demonstrations.
                    # When empty (the normal sample() path) this is a no-op.
                    sample_text = examples_prefix
                    sample_values = {}

                    # For each feature
                    for feature in feature_names:
                        # Create prompt with feature name
                        prompt = f"{sample_text}{feature} is"

                        # Generate only the value (not the next feature name)
                        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                        prompt_length = inputs["input_ids"].shape[1]

                        # Build logits processor for constrained features
                        logits_processor = None
                        if feature in constraint_tries:
                            processor = ConstrainedValueProcessor(
                                trie=constraint_tries[feature],
                                tokenizer=self.tokenizer,
                                prompt_length=prompt_length,
                                first_token_bias=first_token_biases.get(feature),
                            )
                            logits_processor = LogitsProcessorList([processor])

                        # Generate until semicolon or max_length
                        try:
                            # Check if semicolon is in vocabulary
                            semicolon_token = self.tokenizer.encode(";")[0] if ";" in self.tokenizer.decode(list(range(1000))) else None

                            generate_kwargs = dict(
                                max_length=prompt_length + 30,
                                temperature=temperature,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=semicolon_token,
                                do_sample=True,
                            )
                            if logits_processor is not None:
                                generate_kwargs["logits_processor"] = logits_processor

                            output = self.model.generate(
                                inputs["input_ids"],
                                **generate_kwargs,
                            )
                        except (ValueError, RuntimeError, IndexError) as e:
                            # If semicolon token doesn't work, generate with length limit
                            logging.debug(f"Semicolon-based generation failed, falling back to length limit: {e}")
                            fallback_kwargs = dict(
                                max_length=prompt_length + 30,
                                temperature=temperature,
                                pad_token_id=self.tokenizer.eos_token_id,
                                do_sample=True,
                            )
                            if logits_processor is not None:
                                fallback_kwargs["logits_processor"] = logits_processor

                            output = self.model.generate(
                                inputs["input_ids"],
                                **fallback_kwargs,
                            )

                        # Extract the generated value
                        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        raw_value = generated_text[len(prompt):].strip()

                        # Truncate at the EARLIEST delimiter (not just ';'). Otherwise
                        # values like "Female, age is 73;" keep their trailing garbage
                        # because ';' splits after the comma.
                        # For numerical columns we don't treat '.' as a delimiter
                        # (decimals); for cat columns '.' is a sentence boundary.
                        if feature in self.num_cols:
                            delims = [";", ",", "\n"]
                        else:
                            delims = [";", ",", ".", "\n"]
                        earliest = len(raw_value)
                        for d in delims:
                            idx = raw_value.find(d)
                            if idx != -1 and idx < earliest:
                                earliest = idx
                        value = raw_value[:earliest].strip() if earliest < len(raw_value) else raw_value[:30].strip()

                        # Clean up any trailing non-alphanumeric characters
                        while value and not (value[-1].isalnum() or value[-1] in ['.', '-']):
                            value = value[:-1]

                        if feature in self.num_cols:
                            # Try to extract a number if this is a numerical column
                            numeric_match = re.search(r'-?\d+\.?\d*', value)
                            if numeric_match:
                                value = numeric_match.group(0)
                        elif feature in cat_cols:
                            # For categorical columns, try to match one of the known values
                            if feature in categorical_values:
                                valid_cats = categorical_values[feature]
                                # First try direct match
                                matched = False
                                for cat in valid_cats:
                                    if cat.lower() == value.lower():
                                        value = cat  # Use the proper case from the original
                                        matched = True
                                        break

                                # If no direct match, try to find a valid category in the text
                                if not matched:
                                    for cat in valid_cats:
                                        if cat.lower() in value.lower():
                                            value = cat  # Use the proper case from the original
                                            matched = True
                                            break

                        # Store the value
                        sample_values[feature] = value

                        # Update sample text for context in next iteration
                        sample_text += f"{feature} is {value}; "

                    # Create a dictionary with all features in original order
                    ordered_sample = {feature: sample_values.get(feature, "") for feature in self.columns}
                    synthetic_data.append(ordered_sample)

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    print(f"Error generating sample {i+1}: {str(e)}")
                    continue

        # Convert to DataFrame
        if synthetic_data:
            df = pd.DataFrame(synthetic_data)

            # Convert numerical columns to float if possible
            for col in self.num_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not convert column {col} to numeric: {e}")

            return df.head(n_samples)  # Return exactly n_samples rows
        else:
            print("Failed to generate any valid samples.")
            return pd.DataFrame(columns=self.columns)

    def _legacy_sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "auto",
    ) -> pd.DataFrame:
        """
        Legacy method for generating synthetic tabular data samples.

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
            device (str): Device to use for generation. Defaults to "auto" (picks cuda > mps > cpu).
                Set to "cpu", "cuda", "cuda:N", or "mps" to force a specific device.

        Returns:
            pd.DataFrame: DataFrame containing n_samples rows of generated data.
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self._resolve_device(device)

        # Init list for generated DataFrames
        dfs = []

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            _cnt = 0
            try:
                while n_samples > already_generated:
                    start_tokens = great_start.get_start_tokens(k)
                    start_tokens = torch.tensor(start_tokens).to(self.device)

                    # Build attention mask so the model ignores padding tokens
                    attention_mask = (start_tokens != self.tokenizer.pad_token_id).long()

                    # Generate tokens
                    tokens = self.model.generate(
                        input_ids=start_tokens,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
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
                        print(f"{bcolors.WARNING}Unable to generate samples after {_cnt} attempts.{bcolors.ENDC}")
                        print(f"{bcolors.OKBLUE}To address this issue, consider using guided_sampling=True, which uses a different generation approach that may be more reliable, although it might be much slower.{bcolors.ENDC}")
                        print(f"{bcolors.OKBLUE}Example: model.sample(n_samples=10, guided_sampling=True){bcolors.ENDC}")
                        raise Exception("Breaking the generation loop!")

            except Exception as e:
                print(f"{bcolors.FAIL}An error has occurred: {str(e)}{bcolors.ENDC}")
                print(
                    f"{bcolors.WARNING}To address this issue, consider fine-tuning the GReaT model for a longer period. This can be achieved by increasing the number of epochs.{bcolors.ENDC}"
                )
                print(
                    f"{bcolors.WARNING}Alternatively, you might consider increasing the max_length parameter within the sample function. For example: model.sample(n_samples=10, max_length=2000){bcolors.ENDC}"
                )
                
                # Only suggest guided_sampling if we've tried multiple times without success
                if _cnt > 13 and already_generated == 0:
                    print(
                        f"{bcolors.WARNING}You can also try using guided_sampling=True, which uses a different generation approach that may be more reliable, although it might be slower. For example: model.sample(n_samples=10, guided_sampling=True){bcolors.ENDC}"
                    )
                
                print(
                    f"{bcolors.OKBLUE}If the problem persists despite these adjustments, feel free to raise an issue on our GitHub page at: https://github.com/kathrinse/be_great/issues{bcolors.ENDC}"
                )

        # If we have generated at least some samples, return them
        if dfs:
            df_gen = pd.concat(dfs)
            df_gen = df_gen.reset_index(drop=True)
            return df_gen.head(n_samples)
        else:
            # If we couldn't generate any samples with legacy sampling, suggest trying guided sampling
            print(
                f"{bcolors.WARNING}No samples could be generated. Consider trying guided_sampling=True, which uses a different generation approach that may be more reliable, although it might be slower.{bcolors.ENDC}"
            )
            return pd.DataFrame(columns=self.columns)

    def great_sample(
        self,
        starting_prompts: tp.Union[str, list[str]],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "auto",
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
            device: Defaults to "auto" (picks cuda > mps > cpu). Set to "cpu", "cuda", "cuda:N", or "mps" to force a specific device.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        self._resolve_device(device)
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
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(self.device)
            input_ids = torch.unsqueeze(start_token, 0)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Generate tokens
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
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
        device: str = "auto",
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
            device: Defaults to "auto" (picks cuda > mps > cpu). Set to "cpu", "cuda", "cuda:N", or "mps" to force a specific device.

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """

        # Check DataFrame passed.
        if set(df_miss.columns) != set(self.columns):
            raise ValueError(
                "The column names in the DataFrame passed to impute do not match the columns of the GReaT model."
            )

        self._resolve_device(device)

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
                    starting_prompts = _partial_df_to_prompts(df_curr, self.float_precision)
                    df_curr = self.great_sample(
                        starting_prompts, temperature, max_length, device=device
                    )

                    # Convert numerical values to float, flawed numerical values to NaN
                    for i_num_cols in self.num_cols:
                        df_curr[i_num_cols] = pd.to_numeric(
                            df_curr[i_num_cols], errors="coerce"
                        )
                    df_curr[self.num_cols] = df_curr[self.num_cols].astype(np.float64)

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
        If LoRA fine-tuning was used, saves the adapter weights separately using
        PEFT's native ``save_pretrained`` method so they can be reloaded efficiently.

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

            # torch.device is not JSON serializable
            if "device" in attributes and isinstance(attributes["device"], torch.device):
                attributes["device"] = str(attributes["device"])

            json.dump(attributes, f)

        # Save model weights
        if self._is_peft_model:
            # Save only the LoRA adapter weights (much smaller than full model)
            self.model.save_pretrained(path + "/lora_adapter")
            logging.info(f"LoRA adapter saved to {path}/lora_adapter")
        else:
            torch.save(self.model.state_dict(), fs.open(path + "/model.pt", "wb"))

    def load_finetuned_model(self, path: str):
        """Load fine-tuned model

        Load the weights of a fine-tuned large language model into the GReaT pipeline.
        Supports both full model weights (``.pt`` files) and LoRA adapter directories.

        Args:
            path: Path to the fine-tuned model weights file (``.pt``) **or** to a
                directory containing a saved LoRA adapter (created by
                ``PeftModel.save_pretrained``).
        """
        import os

        # Check if path is a LoRA adapter directory
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "Loading a LoRA adapter requires the 'peft' package. "
                    "Install it with:  pip install peft"
                )
            # If the model is already a PeftModel, load the new adapter weights
            if self._is_peft_model:
                self.model.load_adapter(path, adapter_name="default")
            else:
                self.model = PeftModel.from_pretrained(self.model, path)
            logging.info(f"LoRA adapter loaded from {path}")
        else:
            self.model.load_state_dict(torch.load(fsspec.open(path, "rb")))

    @classmethod
    def load_from_dir(cls, path: str):
        """Load GReaT class

        Load trained GReaT model from directory.  Automatically detects whether
        the model was saved with LoRA adapters or as a full checkpoint.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        import os

        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        assert fs.exists(path), f"Directory {path} does not exist."

        # Load attributes
        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance — do NOT apply LoRA in __init__
        # (we will load the adapter weights directly).
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Restore torch.device if it was serialized as a string
        if isinstance(great.device, str):
            great.device = torch.device(great.device)

        # Load model weights — LoRA adapter or full checkpoint
        lora_adapter_path = path + "/lora_adapter"
        has_lora_adapter = (
            fs.exists(lora_adapter_path)
            and fs.exists(lora_adapter_path + "/adapter_config.json")
        )

        if has_lora_adapter:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "Loading a LoRA model requires the 'peft' package. "
                    "Install it with:  pip install peft"
                )
            great.model = PeftModel.from_pretrained(
                great.model, lora_adapter_path, map_location="cpu"
            )
            logging.info(f"LoRA adapter loaded from {lora_adapter_path}")
        else:
            great.model.load_state_dict(
                torch.load(fs.open(path + "/model.pt", "rb"), map_location="cpu")
            )

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

        # Compute per-column statistics for constrained sampling
        self.col_stats = {}
        for col in self.columns:
            if col in self.num_cols:
                self.col_stats[col] = {
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
            else:
                self.col_stats[col] = {
                    "type": "categorical",
                    "categories": df[col].dropna().unique().astype(str).tolist(),
                }

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
