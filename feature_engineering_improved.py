import pandas as pd

# Load improved preprocessed data
df = pd.read_csv('titanic_preprocessed_improved.csv')

# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Extract title from Name
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Encode Title
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Title'] = le.fit_transform(df['Title'])

# Additional features: Age bins and Fare bins
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Mid', 'High', 'VeryHigh'])

# Encode bins
df['AgeBin'] = le.fit_transform(df['AgeBin'])
df['FareBin'] = le.fit_transform(df['FareBin'])

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Save feature engineered data
df.to_csv('titanic_features_improved.csv', index=False)

print("Improved feature engineering complete. Data saved to titanic_features_improved.csv")