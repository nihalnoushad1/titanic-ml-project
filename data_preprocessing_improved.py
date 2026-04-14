import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('titanic.csv')

# Handle missing values with improvements
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Use mean instead of median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)  # Still drop

# Encode categorical data with one-hot for Embarked
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Save preprocessed data
df.to_csv('titanic_preprocessed_improved.csv', index=False)

print("Improved preprocessing complete. Preprocessed data saved to titanic_preprocessed_improved.csv")