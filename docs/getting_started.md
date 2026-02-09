# Getting Started

### Installation

Install the latest version via pip...

```bash
pip install be-great
```

... or download the source code from GitHub

```bash
git clone https://github.com/tabularis-ai/be_great.git
```

### Requirements

GReaT requires Python 3.9 (or higher) and the following packages:

- datasets >= 2.5.2
- numpy >= 1.23.1
- pandas >= 1.4.4
- scikit_learn >= 1.1.1
- scipy >= 1.9.0
- torch >= 1.10.2
- tqdm >= 4.64.1
- transformers >= 4.22.1
- accelerate >= 0.20.1
- fsspec >= 2024.5.0

**Optional:**

- peft >= 0.14.0 (for LoRA fine-tuning)


### Quickstart

In the example below, we show how the GReaT approach is used to generate synthetic tabular data for the California Housing dataset.
```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=32, epochs=50, fp16=True)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

### Random Preconditioning

During training, GReaT conditions on a single column by default. This can lead to overfitting on that column. Enable random preconditioning to select a different column each epoch:

```python
model.fit(data, random_conditional_col=True)
```

### Guided Sampling & Float Precision

For small datasets or datasets with many features, use guided sampling and limited float precision:

```python
model = GReaT(
    llm='distilgpt2',
    float_precision=3,
    batch_size=8,
    epochs=100,
    fp16=True,
)
model.fit(data)

synthetic_data = model.sample(
    n_samples=100,
    guided_sampling=True,
    random_feature_order=True,
    temperature=0.7,
)
```

### LoRA Fine-Tuning

GReaT supports LoRA for parameter-efficient fine-tuning, reducing memory usage and training time:

```bash
pip install peft
```

```python
model = GReaT(
    llm='distilgpt2',
    batch_size=32,
    epochs=5,
    efficient_finetuning="lora",
    lora_config={"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
    fp16=True,
)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

Target modules are auto-detected for common architectures (GPT-2, LLaMA, Falcon, etc.), or can be specified explicitly via `lora_config["target_modules"]`.

### Evaluating Synthetic Data

After generating synthetic data, use the built-in metrics to measure quality and privacy:

```python
from be_great.metrics import ColumnShapes, DiscriminatorMetric, MLEfficiency, DistanceToClosestRecord
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Compare column distributions (KS test / TVD)
ColumnShapes().compute(data, synthetic_data)

# Train a classifier to distinguish real vs synthetic (0.5 = best)
DiscriminatorMetric().compute(data, synthetic_data)

# Train on synthetic, test on real
MLEfficiency(
    model=RandomForestClassifier,
    metric=accuracy_score,
    model_params={"n_estimators": 100},
).compute(data, synthetic_data, label_col="target")

# Check privacy: distance to closest real record
DistanceToClosestRecord().compute(data, synthetic_data)
```

See Examples to find more details.
