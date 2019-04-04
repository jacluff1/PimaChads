import pandas as pd

filename =

df = pd.read_csv(filename)

# Number of observations.
N = df.shape[0]

fractions = []

for col in df.columns:
    # Get categorical columns.
    if set(df[col]) <= {0, 1}:
        # (column name, amount true)
        fractions.append((col, sum(df[col]) / N))

fractions.sort()

print(f"(column name, amount True) = {fractions}")
