import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load improved feature engineered data
df = pd.read_csv('titanic_features_improved.csv')

# Prepare data
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train improved model: RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()}")

# Evaluate on train
y_pred = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred)}")

# Save model
joblib.dump(model, 'titanic_model_improved.pkl')

print("Improved model training complete. Model saved to titanic_model_improved.pkl")