<!-- markdownlint-disable -->

<a href="https://github.com/tabularis-ai/be_great/tree/main/be_great/great.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `great`

---

## <kbd>class</kbd> `GReaT`
GReaT Class

The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data, and to sample synthetic tabular data.

**Attributes:**

 - <b>`llm`</b> (str):  [HuggingFace checkpoint](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) of a pretrained large language model, used as basis of our model
 - <b>`tokenizer`</b> (AutoTokenizer):  Tokenizer, automatically downloaded from llm-checkpoint
 - <b>`model`</b> (AutoModelForCausalLM):  Large language model, automatically downloaded from llm-checkpoint
 - <b>`experiment_dir`</b> (str):  Directory, where the training checkpoints will be saved
 - <b>`epochs`</b> (int):  Number of epochs to fine-tune the model
 - <b>`batch_size`</b> (int):  Batch size used for fine-tuning
 - <b>`efficient_finetuning`</b> (str):  Fine-tuning method. Set to `"lora"` for LoRA fine-tuning.
 - <b>`float_precision`</b> (int | None):  Number of decimal places for floating point values. None means full precision.
 - <b>`train_hyperparameters`</b> (dict):  Additional hyperparameters added to the TrainingArguments used by the HuggingFace Library, see here the [full list](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) of all possible values
 - <b>`columns`</b> (list):  List of all features/columns of the tabular dataset
 - <b>`num_cols`</b> (list):  List of all numerical features/columns of the tabular dataset
 - <b>`conditional_col`</b> (str):  Name of a feature/column on which the sampling can be conditioned
 - <b>`conditional_col_dist`</b> (dict | list):  Distribution of the feature/column specified by conditional_col

---

### <kbd>method</kbd> `GReaT.__init__`

```python
__init__(
    llm: str,
    experiment_dir: str = 'trainer_great',
    epochs: int = 100,
    batch_size: int = 8,
    efficient_finetuning: str = '',
    lora_config: Optional[Dict[str, Any]] = None,
    float_precision: Optional[int] = None,
    report_to: List[str] = [],
    **train_kwargs
)
```

Initializes GReaT.

**Args:**

 - <b>`llm`</b>:  [HuggingFace checkpoint](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) of a pretrained large language model, used as basis for our model
 - <b>`experiment_dir`</b>:  Directory, where the training checkpoints will be saved
 - <b>`epochs`</b>:  Number of epochs to fine-tune the model
 - <b>`batch_size`</b>:  Batch size used for fine-tuning
 - <b>`efficient_finetuning`</b>:  Fine-tuning method. Set to `"lora"` to enable LoRA (Low-Rank Adaptation) fine-tuning. Requires the `peft` package.
 - <b>`lora_config`</b>:  Optional dictionary of LoRA hyperparameters to override defaults. Supported keys: `r` (rank, default 16), `lora_alpha` (scaling factor, default 32), `target_modules` (list of module names or None for auto-detection), `lora_dropout` (default 0.05), `bias` (default "none"), `task_type` (default "CAUSAL_LM"), `modules_to_save` (default None).
 - <b>`float_precision`</b>:  Number of decimal places to use for floating point numbers. If None, full precision is used.
 - <b>`report_to`</b>:  List of integrations to report to (e.g. `["wandb"]`). Empty list disables reporting.
 - <b>`train_kwargs`</b>:  Additional hyperparameters added to the TrainingArguments used by the HuggingFace Library, see here the [full list](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) of all possible values

---

### <kbd>method</kbd> `GReaT.fit`

```python
fit(
    data: Union[DataFrame, ndarray],
    column_names: Optional[List[str]] = None,
    conditional_col: Optional[str] = None,
    resume_from_checkpoint: Union[bool, str] = False,
    random_conditional_col: bool = True
) → GReaTTrainer
```

Fine-tune GReaT using tabular data.

**Args:**

 - <b>`data`</b>:  Pandas DataFrame or Numpy Array that contains the tabular data
 - <b>`column_names`</b>:  If data is Numpy Array, the feature names have to be defined. If data is Pandas DataFrame, the value is ignored
 - <b>`conditional_col`</b>:  If given, the distribution of this column is saved and used as a starting point for the generation process later. If None, the last column is considered as conditional feature
 - <b>`resume_from_checkpoint`</b>:  If True, resumes training from the latest checkpoint in the experiment_dir. If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)
 - <b>`random_conditional_col`</b>:  If True, a different random column is selected for conditioning at the end of each training epoch. This prevents overfitting on a single column and leads to more balanced synthetic data.

**Returns:**
 GReaTTrainer used for the fine-tuning process

---

### <kbd>method</kbd> `GReaT.sample`

```python
sample(
    n_samples: int,
    start_col: Optional[str] = '',
    start_col_dist: Optional[Union[dict, list]] = None,
    temperature: float = 0.7,
    k: int = 100,
    max_length: int = 100,
    drop_nan: bool = False,
    device: str = 'cuda',
    guided_sampling: bool = False,
    random_feature_order: bool = True
) → DataFrame
```

Generate synthetic tabular data samples.

**Args:**

 - <b>`n_samples`</b>:  Number of synthetic samples to generate
 - <b>`start_col`</b>:  Feature to use as starting point for the generation process. If not given, the target learned during the fitting is used as starting point
 - <b>`start_col_dist`</b>:  Feature distribution of the starting feature. Should have the format `{"F1": p1, "F2": p2, ...}` for discrete columns or be a list of possible values for continuous columns. If not given, the target distribution learned during the fitting is used as starting point
 - <b>`temperature`</b>:  The generation samples each token from the probability distribution given by a softmax function. The temperature parameter controls the softmax function. A low temperature makes it sharper (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. See this [blog article](https://huggingface.co/blog/how-to-generate) to read more about the generation process.
 - <b>`k`</b>:  Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
 - <b>`max_length`</b>:  Maximal number of tokens to generate - has to be long enough to not cut any information!
 - <b>`drop_nan`</b>:  If True, rows with any NaN values are dropped from the generated output
 - <b>`device`</b>:  Set to `"cpu"` if the GPU should not be used. You can also specify the concrete GPU (e.g. `"cuda:0"`)
 - <b>`guided_sampling`</b>:  If True, enables feature-by-feature guided generation. This is slower but can produce more reliable results for datasets with many features or complex relationships.
 - <b>`random_feature_order`</b>:  If True (and `guided_sampling=True`), the order of feature generation is randomized for each sample. Helps avoid ordering bias.

**Returns:**
 Pandas DataFrame with n_samples rows of generated data

---

### <kbd>method</kbd> `GReaT.great_sample`

```python
great_sample(
    starting_prompts: Union[str, list[str]],
    temperature: float = 0.7,
    max_length: int = 100,
    device: str = 'cuda'
) → DataFrame
```

Generate synthetic tabular data samples conditioned on a given input.

**Args:**

 - <b>`starting_prompts`</b>:  String or List of Strings on which the output is conditioned. For example, `"Sex is female, Age is 26"`
 - <b>`temperature`</b>:  The generation samples each token from the probability distribution given by a softmax function. The temperature parameter controls the softmax function. A low temperature makes it sharper (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output. See this [blog article](https://huggingface.co/blog/how-to-generate) to read more about the generation process.
 - <b>`max_length`</b>:  Maximal number of tokens to generate - has to be long enough to not cut any information
 - <b>`device`</b>:  Set to `"cpu"` if the GPU should not be used. You can also specify the concrete GPU.

**Returns:**
 Pandas DataFrame with synthetic data generated based on starting_prompts

---

### <kbd>method</kbd> `GReaT.impute`

```python
impute(
    df_miss: DataFrame,
    temperature: float = 0.7,
    k: int = 100,
    max_length: int = 100,
    max_retries: int = 15,
    device: str = 'cuda'
) → DataFrame
```

Impute a DataFrame with missing values using a trained GReaT model.

**Args:**

 - <b>`df_miss`</b>:  Pandas DataFrame of the exact same format (column names, value ranges/types) as the data used to train the GReaT model, with missing values indicated by NaN. This function will sample the missing values conditioned on the remaining values.
 - <b>`temperature`</b>:  Controls the softmax function during generation. Lower values produce more deterministic output.
 - <b>`k`</b>:  Sampling batch size
 - <b>`max_length`</b>:  Maximal number of tokens to generate
 - <b>`max_retries`</b>:  Maximum number of retries if imputation fails to fill all values
 - <b>`device`</b>:  Set to `"cpu"` if the GPU should not be used

**Returns:**
 Pandas DataFrame with imputed values

---

### <kbd>method</kbd> `GReaT.save`

```python
save(path: str)
```

Save GReaT Model

Saves the model weights and a configuration file in the given directory. If LoRA fine-tuning was used, saves the adapter weights separately using PEFT's native `save_pretrained` method so they can be reloaded efficiently. Supports remote file systems via `fsspec` (e.g. `s3://`, `gs://`).

**Args:**

 - <b>`path`</b>:  Path where to save the model

---

### <kbd>classmethod</kbd> `GReaT.load_from_dir`

```python
load_from_dir(path: str)
```

Load GReaT class

Load trained GReaT model from directory. Automatically detects whether the model was saved with LoRA adapters or as a full checkpoint. Supports remote file systems via `fsspec`.

**Args:**

 - <b>`path`</b>:  Directory where GReaT model is saved

**Returns:**
 New instance of GReaT loaded from directory

---

_This file was manually updated to match the current source code._
