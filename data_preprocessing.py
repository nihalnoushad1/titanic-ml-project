import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('titanic.csv')


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)  # Drop Cabin due to many missing

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

df.to_csv('titanic_preprocessed.csv', index=False)

print("Preprocessing complete. Preprocessed data saved to titanic_preprocessed.csv")
