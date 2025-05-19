#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of using GReaT with random preconditioning feature.

This example demonstrates how to use the random preconditioning feature in GReaT
to improve the quality of synthetic data generation. The random preconditioning
feature selects a different random column for conditioning in each epoch during training.

This is useful to prevent any single column from being overfitted during training,
resulting in more balanced synthetic data across all columns.
"""

import pandas as pd
import numpy as np
from be_great import GReaT
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True).frame

# Split the data into training and test sets for evaluation
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Configure GReaT model with random preconditioning
print("Initializing GReaT model...")
model = GReaT(
    llm='distilgpt2',
    batch_size=32,
    epochs=10,  # Reduced for example purposes
    fp16=True,
    experiment_dir="random_preconditioning_example"
)

# Train the model with random preconditioning enabled
print("Training model with random preconditioning...")
model.fit(train_data, random_conditional_col=True)

# Generate synthetic data using the trained model
print("Generating synthetic data...")
synthetic_data = model.sample(n_samples=len(test_data))

# Print statistical comparison between original and synthetic data
print("\nComparison of mean values:")
print("---------------------------")
print("Column               | Original | Synthetic")
print("-" * 50)
for col in data.columns:
    orig_mean = test_data[col].mean()
    synth_mean = synthetic_data[col].mean()
    print(f"{col:20} | {orig_mean:.4f} | {synth_mean:.4f}")

print("\nComparison of standard deviations:")
print("----------------------------------")
print("Column               | Original | Synthetic")
print("-" * 50)
for col in data.columns:
    orig_std = test_data[col].std()
    synth_std = synthetic_data[col].std()
    print(f"{col:20} | {orig_std:.4f} | {synth_std:.4f}")

# Save the model and the synthetic data
print("\nSaving model and synthetic data...")
model.save("random_preconditioning_model")
synthetic_data.to_csv("random_preconditioning_synthetic_data.csv", index=False)

print("\nDone! Model saved to 'random_preconditioning_model' and synthetic data saved to 'random_preconditioning_synthetic_data.csv'") 