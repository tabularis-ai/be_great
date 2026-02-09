"""Conditional sampling example using the UCI Adult dataset."""

import pandas as pd
from ucimlrepo import fetch_ucirepo
from be_great import GReaT

# --- Load & prep data ---
adult = fetch_ucirepo(id=2)
df = adult.data.features.copy()
df["income"] = adult.data.targets["income"]

cols = ["age", "workclass", "education", "occupation", "sex", "hours-per-week", "income"]
df = df[cols]
df = df[~df.isin(["?"]).any(axis=1)].dropna()
df["income"] = df["income"].str.replace(".", "", regex=False)
df = df.sample(n=2000, random_state=42).reset_index(drop=True)

# --- Train ---
great = GReaT("distilgpt2", epochs=50, batch_size=32, float_precision=0)
great.fit(df)

# --- Unconstrained baseline ---
samples = great.sample(n_samples=50, guided_sampling=True, device="cuda")
print("=== Unconstrained ===")
print(samples[["age", "sex", "hours-per-week"]].describe())

# --- Single numeric constraint: age >= 50 ---
samples_old = great.sample(n_samples=50, conditions={"age": ">= 50"}, device="cuda")
print("\n=== age >= 50 ===")
print(f"age range: {samples_old['age'].min()} - {samples_old['age'].max()}")
assert (samples_old["age"] >= 50).all()

# --- Categorical constraint: only Female ---
samples_f = great.sample(n_samples=50, conditions={"sex": "== 'Female'"}, device="cuda")
print("\n=== sex == Female ===")
print(samples_f["sex"].value_counts())

# --- Multiple constraints ---
samples_multi = great.sample(
    n_samples=50,
    conditions={"age": ">= 40", "hours-per-week": "<= 40", "sex": "!= 'Male'"},
    device="cuda",
)
print("\n=== age>=40 & hours<=40 & sex!=Male ===")
print(f"age min:   {samples_multi['age'].min()}")
print(f"hours max: {samples_multi['hours-per-week'].max()}")
print(f"sex vals:  {samples_multi['sex'].unique().tolist()}")
assert (samples_multi["age"] >= 40).all()
assert (samples_multi["hours-per-week"] <= 40).all()
assert (samples_multi["sex"] != "Male").all()
print("\nAll constraints satisfied!")
