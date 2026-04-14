import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv('titanic_features.csv')


X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred)}")

joblib.dump(model, 'titanic_model.pkl')

print("Model training complete. Model saved to titanic_model.pkl")
