import pandas as pd

df = pd.read_csv('titanic_preprocessed.csv')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)


df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Title'] = le.fit_transform(df['Title'])

df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


df.to_csv('titanic_features.csv', index=False)

print("Feature engineering complete. Data saved to titanic_features.csv")
