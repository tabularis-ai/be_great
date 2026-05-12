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

### Live Quality Monitoring During Training

Pass a held-out `eval_data` DataFrame to `fit()` to monitor synthesis quality **while training is still running**. After each epoch (or every N steps), GReaT samples from the in-training model, compares the sample against `eval_data` using `ColumnShapes` similarity (KS test for numerical columns, Total Variation Distance for categorical), and shows the score live in the [chugchug](https://github.com/unnir/chugchug) progress bar as `q=…`. A printed line `🎯 [quality @ step N] column_shapes_mean = 0.XXXX` is also emitted after every evaluation.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from be_great import GReaT

df = load_iris(as_frame=True).frame
train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

model = GReaT(llm="distilgpt2", epochs=5, batch_size=8)
model.fit(
    train_df,
    eval_data=eval_df,        # held-out comparison set
    eval_n_samples=50,        # synth rows drawn per evaluation
    eval_every=None,          # None = once per epoch; int = every N training steps
)
```

Score interpretation: `column_shapes_mean` is in `[0, 1]`. **1.0** means the synthetic and real marginal distributions are indistinguishable; values close to 0 mean very different distributions. Watch it climb across epochs — that's your training signal beyond raw loss.

Tips:
- Sampling during evaluation is the cost driver. Keep `eval_n_samples` small (20–50) for fast feedback, larger (100–200) for a more stable score.
- The model is automatically set to `eval()` during evaluation and restored to `train()` after; the optimizer is not disturbed.
- On Apple Silicon (MPS) the model stays on MPS during eval — no device thrashing.

### Auto Device Selection (CUDA / MPS / CPU)

`sample()`, `impute()`, and all generation methods default to `device="auto"`, which picks CUDA when available, then Apple's MPS, then CPU. You can still pin a specific device when you need to: `model.sample(n_samples=100, device="cpu")` or `device="cuda:1"`.

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
