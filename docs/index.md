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

### Saving and Loading

GReaT also allows you to save your trained model for later use:

```python
# Save
model.save("my_directory")  # saves a "model.pt" and a "config.json" file

# Load
model = GReaT.load_from_dir("my_directory")  # loads the model again
```


