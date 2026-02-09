<p align="center">
<img src="https://github.com/kathrinse/be_great/raw/main/imgs/GReaT_logo.png" width="326"/>
</p>

<p align="center">
<strong>Generation of Realistic Tabular data</strong>
<br> with pretrained Transformer-based language models
</p>

# GReaT

GReaT is a simple to use framework to generate tabular data with transformer-based language models.

## Installation

```bash
pip install be-great
```

## Usage with Quickstart

For a more complete introduction we provide example Jupyter notebooks.

Example: (California Housing dataset)

```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(
    llm='distilgpt2',    # Language model
    batch_size=32,       # How many samples are processed at once during training
    epochs=50,           # Total number of training epochs
    fp16=True,           # Enable faster mixed precision training
    dataloader_num_workers=4 # How much CPU processor to use during training 
)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

## Advanced Features

### Random Preconditioning

During training, GReaT uses a column of the dataset to condition the model. By default, it uses the last column of the dataset. However, this can lead to the model overfitting on that specific column, resulting in synthetic data where that column is almost identical to the original data, while other columns show more variability.

To prevent this issue, you can enable random preconditioning, which selects a different random column for conditioning in each epoch during training:

```python
# Enable random preconditioning during training
model.fit(data, random_conditional_col=True)
```

This typically leads to more balanced synthetic data, where all columns maintain appropriate variability compared to the original dataset.

### Imputing a sample

GReaT provides an imputation feature to fill in missing values in a dataframe:

```python
# Randomly drop 50% of the values in the test data
import numpy as np
for clm in test_data.columns:
    test_data[clm]=test_data[clm].apply(lambda x: (x if np.random.rand() > 0.5 else np.nan))

# Impute the missing values
imputed_data = model.impute(test_data, max_length=200)
```

### Optimizing for Challenging Datasets

When working with small datasets or datasets with many features, GReaT offers specialized parameters to improve generation quality:

```python
model = GReaT(
    llm='distilgpt2',
    float_precision=3,           # Limit floating-point precision to 3 decimal places
    batch_size=8,
    epochs=100,
    fp16=True,
)
model.fit(data)

# Use guided sampling for higher quality generation with complex feature sets
synthetic_data = model.sample(
    n_samples=100,
    guided_sampling=True,        # Enable feature-by-feature guided generation
    random_feature_order=True,   # Randomize feature order to avoid bias
    temperature=0.7,
)
```

The `guided_sampling=True` parameter enables a feature-by-feature generation approach, which can produce more reliable results for datasets with many features or complex relationships.

The `float_precision` parameter limits decimal places in numerical values, which can help the model focus on significant patterns rather than memorizing exact values.

### Efficient Fine-Tuning with LoRA

GReaT supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. This drastically reduces memory usage and training time, making it possible to fine-tune larger models on consumer hardware.

```bash
pip install peft
```

```python
# LoRA with auto-detected target modules (works across model architectures)
model = GReaT(
    llm='meta-llama/Llama-3.1-8B-Instruct',
    batch_size=32,
    epochs=5,
    efficient_finetuning="lora",
    fp16=True,
)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

You can also customize the LoRA hyperparameters:

```python
model = GReaT(
    llm='distilgpt2',
    batch_size=32,
    epochs=5,
    efficient_finetuning="lora",
    lora_config={
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],  # optional, auto-detected if omitted
    },
    fp16=True,
)
model.fit(data)
```

### Saving and Loading

GReaT also allows you to save your trained model for later use, including LoRA models:

```python
# Save
model.save("my_directory")  # saves a "model.pt" and a "config.json" file

# Load
model = GReaT.load_from_dir("my_directory")  # loads the model again

# Supports remote file systems via fsspec
model.save("s3://my_bucket")
model = GReaT.load_from_dir("s3://my_bucket")
```

### Evaluating Synthetic Data

GReaT includes a built-in metrics suite to evaluate the quality, utility, and privacy of generated data. All metrics share the same interface:

```python
from be_great.metrics import ColumnShapes, DiscriminatorMetric, DistanceToClosestRecord

# Statistical: per-column distribution similarity
ColumnShapes().compute(real_data, synthetic_data)

# Fidelity: can a classifier tell real from synthetic?
DiscriminatorMetric().compute(real_data, synthetic_data)

# Privacy: distance from synthetic records to nearest real neighbor
DistanceToClosestRecord().compute(real_data, synthetic_data)
```

See the [Metrics API Reference](./api-docs/metrics.md) for the full list of available metrics.

