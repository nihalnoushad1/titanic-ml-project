import pandas as pd

# Load the dataset
df = pd.read_csv('titanic.csv')

# Inspect structure and basic details
print("First 5 rows:")
print(df.head())
print("\nData types and non-null counts:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())