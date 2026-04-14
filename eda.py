import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('titanic_preprocessed.csv')


df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig('distributions.png')

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[float, int])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.savefig('correlation_heatmap.png')


sns.barplot(x='Sex', y='Survived', data=df)
plt.savefig('survival_by_sex.png')

print("EDA complete. Plots saved.")
