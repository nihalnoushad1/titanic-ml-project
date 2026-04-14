import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load improved preprocessed data
df = pd.read_csv('titanic_preprocessed_improved.csv')

# Analyze distributions
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig('distributions_improved.png')

# Analyze relationships
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[float, int])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.savefig('correlation_heatmap_improved.png')

# Survival by sex
sns.barplot(x='Sex', y='Survived', data=df)
plt.savefig('survival_by_sex_improved.png')

# Additional: Pairplot for key features
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.savefig('pairplot_improved.png')

print("Improved EDA complete. Plots saved.")