
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Wczytanie danych
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Wczytanie transformatora
preprocessor = joblib.load('preprocessor.pkl')

# Przetwarzanie danych
X_train_transformed = preprocessor.transform(X_train)

# Trenowanie modelu
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Zapis modelu
joblib.dump(model, 'model.pkl')
