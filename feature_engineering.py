import pandas as pd

# Load preprocessed data
df = pd.read_csv('titanic_preprocessed.csv')

# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Extract title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Encode Title
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Title'] = le.fit_transform(df['Title'])

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Save feature engineered data
df.to_csv('titanic_features.csv', index=False)

print("Feature engineering complete. Data saved to titanic_features.csv")