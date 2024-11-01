
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Wczytanie danych
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Wczytanie transformatora i modelu
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# Przetwarzanie danych testowych
X_test_transformed = preprocessor.transform(X_test)

# Predykcja
y_pred = model.predict(X_test_transformed)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Wykres przewidywane vs rzeczywiste
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Predicted vs Actual Values")
plt.savefig('pred_vs_actual.png')
plt.close()
