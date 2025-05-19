[![PyPI version](https://badge.fury.io/py/be-great.svg)](https://badge.fury.io/py/be-great) [![Downloads](https://static.pepy.tech/badge/be-great)](https://pepy.tech/project/be-great)

[//]: # (![Screenshot]&#40;https://github.com/kathrinse/be_great/blob/main/imgs/GReaT_logo.png&#41;)
<p align="center">
<img src="https://github.com/kathrinse/be_great/raw/main/imgs/GReaT_logo.png" width="326"/>
</p>

<p align="center">
<strong>Generation of Realistic Tabular data</strong>
<br> with pretrained Transformer-based language models
</p>

&nbsp;
&nbsp;
&nbsp;

Our GReaT framework leverages the power of advanced pretrained Transformer language models to produce high-quality synthetic tabular data. Generate new data samples effortlessly with our user-friendly API in just a few lines of code. Please see our [publication](https://openreview.net/forum?id=cEygmQNOeI) for more details. 

## GReaT Installation

The GReaT framework can be easily installed using with [pip](https://pypi.org/project/pip/) - requires a Python version >= 3.9: 
```bash
pip install be-great
```



## GReaT Quickstart

In the example below, we show how the GReaT approach is used to generate synthetic tabular data for the California Housing dataset.
```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=32,  epochs=50,
              fp16=True, dataloader_num_workers=4)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kathrinse/be_great/blob/main/examples/GReaT_colab_example.ipynb)

### Imputing a sample
GReaT also features an interface to impute, i.e., fill in, missing values in arbitrary combinations. This requires a trained ``model``, for instance one obtained using the code snippet above, and a ```pd.DataFrame``` where missing values are set to NaN.
A minimal example is provided below:
```python
# test_data: pd.DataFrame with samples from the distribution
# model: GReaT trained on the data distribution that should be imputed

# Drop values randomly from test_data
import numpy as np
for clm in test_data.columns:
    test_data[clm]=test_data[clm].apply(lambda x: (x if np.random.rand() > 0.5 else np.nan))

imputed_data = model.impute(test_data, max_length=200)
```

### Saving and Loading
GReaT provides methods for saving a model checkpoint (besides the checkpoints stored by the huggingface transformers Trainer) and loading the checkpoint again.
```python
model = GReaT(llm='distilgpt2', batch_size=32,  epochs=50, fp16=True)
model.fit(data)
model.save("my_directory")  # saves a "model.pt" and a "config.json" file
model = GReaT.load_from_dir("my_directory")  # loads the model again

# supports remote file systems via fsspec
model.save("s3://my_bucket")
model = GReaT.load_from_dir("s3://my_bucket")
```

## Optimizing GReaT for Challenging Datasets

When working with small datasets or datasets with many features, GReaT offers specialized parameters to improve generation quality:

```python
# For small datasets or datasets with many features
model = GReaT(
    llm='distilgpt2',
    float_precision=3,  # Limit floating-point precision to 3 decimal places
    batch_size=8,       # Use smaller batch size for small datasets
    epochs=100,         # Train for more epochs with small data
    fp16=True           # Enable half-precision training for faster computation and lower memory usage
)
model.fit(data)

# Use guided sampling for higher quality generation with complex feature sets
synthetic_data = model.sample(
    n_samples=100,
    guided_sampling=True,     # Enable feature-by-feature guided generation
    random_feature_order=True,  # Randomize feature order to avoid bias
    temperature=0.7           # Control diversity of generated values
)
```

The `guided_sampling=True` parameter enables a feature-by-feature generation approach, which can produce more reliable results for datasets with many features or complex relationships. While potentially slower than the default sampling method, it can help overcome generation challenges with difficult datasets.

The `float_precision` parameter limits decimal places in numerical values, which can help the model focus on significant patterns rather than memorizing exact values. This is particularly helpful for small datasets where overfitting is a concern.

## GReaT Citation 

If you use GReaT, please link or cite our work:

``` bibtex
@inproceedings{borisov2023language,
  title={Language Models are Realistic Tabular Data Generators},
  author={Vadim Borisov and Kathrin Sessler and Tobias Leemann and Martin Pawelczyk and Gjergji Kasneci},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=cEygmQNOeI}
}
```

## Custom Synthetic Data

Need synthetic data for your business? We can help!
Contact us at info@tabularis.ai for custom data generation services.
