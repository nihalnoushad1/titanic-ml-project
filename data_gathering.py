import pandas as pd


df = pd.read_csv('titanic.csv')

print("First 5 rows:")
print(df.head())
print("\nData types and non-null counts:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())
