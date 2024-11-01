import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

def main():
    cleaned_data_path = 'cleaned_data.csv'

    try:
        data = pd.read_csv(cleaned_data_path)
        print('Cleaned data loaded successfully.')
    except Exception as e:
        print(f'Error loading cleaned data: {e}')
        return

    categorical_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'region', 'education', 'income']

    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    if 'score' not in data.columns:
        print("'score' column not found in data.")
        return

    X = data.drop(['score', 'rownames'], axis=1, errors='ignore')
    y = data['score']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f'Error splitting data: {e}')
        return

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("Model Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

if __name__ == '__main__':
    main()