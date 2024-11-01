
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Wczytanie danych
data = pd.read_csv('CollegeDistance.csv')

# Oddzielenie zmiennej celu
X = data.drop(columns=['score', 'rownames'])
y = data['score']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inżynieria cech
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Dopasowanie transformatora i zapisanie go
preprocessor.fit(X_train)
joblib.dump(preprocessor, 'preprocessor.pkl')

# Zapis zbiorów treningowych i testowych
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
