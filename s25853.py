# s25853.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Generowanie prostego zbioru danych
def generate_data():
    np.random.seed(42)

    # Generowanie pierwszej chmury punktów
    cloud1 = np.random.randn(100, 2) + np.array([5, 5])

    # Generowanie drugiej chmury punktów
    cloud2 = np.random.randn(100, 2) + np.array([-5, -5])

    # Łączenie danych
    X = np.vstack([cloud1, cloud2])
    y = np.hstack([np.zeros(cloud1.shape[0]), np.ones(cloud2.shape[0])])  # 0 dla cloud1, 1 dla cloud2

    return X, y


# Trenowanie prostego modelu regresji logistycznej
def train_model():
    # Generowanie danych
    X, y = generate_data()

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)

    # Wyliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)

    # Zapis wyniku do pliku accuracy.txt
    with open('accuracy.txt', 'w') as f:
        f.write(f"Model trained with accuracy: {accuracy * 100:.2f}%\n")

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_model()
