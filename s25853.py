import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
import argparse

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

# Dane Google API
SERVICE_ACCOUNT_FILE = 'creds.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = '1MsvM9xTRq2FEyA0UmFihTUp-XEJ5aHt_GORBFCdQbXQ'
RANGE_NAME = 'Arkusz1!A1:Z'  # Zakres, w którym są dane

# Autoryzacja z użyciem service account
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=credentials)


# Odczyt danych z csv
def read_data_from_csv(file_path):
    logger.info("Odczyt danych z pliku csv...")

    df = pd.read_csv(file_path, sep=',')  # Załóżmy, że plik jest oddzielany tabulatorami

    logger.info("Dane odczytane pomyślnie.")

    # Zastąpienie wartości NaN pustymi stringami
    df.fillna("", inplace=True)
    return df


# Pobierz dane z Google Sheets
def get_data_from_sheets():
    logger.info("Odczyt danych z Google Sheets...")

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])
    if not values:
        logger.error("Brak danych")
        return None
    # Konwersja danych do DataFrame
    df = pd.DataFrame(values[1:], columns=values[0])  # Nagłówki z pierwszego wiersza

    logger.info("Dane odczytane pomyślnie.")

    return df


# Czyszczenie arkusza Google Sheets
def clear_google_sheet():
    sheet = service.spreadsheets()
    # Wykonaj czyszczenie na tym samym zakresie, co aktualizacja
    result = sheet.values().clear(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    logger.info("Czyszczenie Google Sheets.")


# Aktualizowanie arkusza Google Sheets
def update_google_sheet(df_to_update):
    logger.info("Aktualizowanie arkusza Google Sheets")
    clear_google_sheet()

    sheet = service.spreadsheets()
    body = {
        'values': [df_to_update.columns.tolist()] + df_to_update.values.tolist()  # Dodaj nagłówki
    }

    result = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
        valueInputOption='RAW', body=body).execute()

    logger.info("Dane zaktualizowane pomyślnie.")


# Czyszczenie danych
def clean_data(df_to_clean, missing_threshold=0.5):
    logger.info("Rozpoczęcie czyszczenia danych...")
    # Zamiana pustych stringów na NaN
    df_to_clean.replace("", np.nan, inplace=True)

    # Usuwanie wierszy z zbyt wieloma brakującymi wartościami
    threshold = len(df_to_clean.columns) * (1 - missing_threshold)
    df_cleaned = df_to_clean[df_to_clean.notnull().sum(axis=1) >= threshold]

    # wartości puste przed uzupełnieniem braków
    empty_before = df_to_clean.isnull().sum().sum()

    # Uzupełnianie braków
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.int64, np.float64]:  # Kolumny numeryczne
            median_value = df_cleaned[column].median()  # Mediana
            df_cleaned.loc[:, column] = df_cleaned[column].fillna(median_value)  # Użyj .loc[]
        else:  # Kolumny nienumeryczne
            df_cleaned.loc[:, column] = df_cleaned[column].fillna("Brak danych")  # Użyj .loc[]

    # Zastąpienie wartości NaN pustymi stringami - potrzebne w przypadku nie uzupełnienia braków
    # df_cleaned.fillna("", inplace=True)
    logger.info("Pomyślne zakończenie czysczenia danych.")

    return df_cleaned, empty_before


# Standaryzacja danych (numeryczne i nienumeryczne)
def standardize_data(df_to_standardize):
    logger.info("Rozpoczęcie standaryzacji danych...")

    # Tworzymy kopię DataFrame, żeby nie modyfikować oryginalnych danych
    df_to_standardized = df_to_standardize.copy()

    # Zakodowanie kolumn nienumerycznych na wartości liczbowe
    label_encoders = {}  # Przechowywanie encoderów dla każdej kolumny
    for column in df_to_standardized.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_to_standardized[column] = le.fit_transform(df_to_standardized[column])
        label_encoders[column] = le  # Zachowaj encoder, jeśli chcesz później wrócić do oryginalnych wartości

    # Wybieramy wszystkie kolumny do standaryzacji (teraz są wszystkie numeryczne)
    numerical_cols = df_to_standardized.columns.tolist()

    # Tworzymy instancję StandardScaler
    scaler = StandardScaler()

    # Standaryzujemy dane
    df_to_standardized[numerical_cols] = scaler.fit_transform(df_to_standardized[numerical_cols])

    logger.info("Pomyślne zakończenie standaryzacji danych.")

    return df_to_standardized


# Główna funkcja
def main():
    # Definiowanie argumentów wejściowych
    parser = argparse.ArgumentParser(
        description='Opcje: --upload (upload danych do Google Sheets) --standardize (czyszczenie i standaryzacja danych)')
    parser.add_argument('--upload', type=str, help='Ścieżka do pliku CSV do wczytania i uploadu do Google Sheets')
    parser.add_argument('--standardize', action='store_true',
                        help='Ścieżka do pliku CSV do czyszczenia i standaryzacji danych')

    args = parser.parse_args()

    # Obsługa opcji --upload
    if args.upload:
        df = read_data_from_csv(args.upload)
        update_google_sheet(df)

    # Obsługa opcji --standardize
    elif args.standardize:
        df = get_data_from_sheets()
        if df is None:
            logger.error("Błąd przy odczycie danych z Google Sheets.")
            return

        # Obliczenie zmienionych rekordów
        rows_before = df.shape[0]
        df_cleaned, empty_before = clean_data(df)
        rows_after = df_cleaned.shape[0]
        empty_after = df_cleaned.isnull().sum().sum()

        changed_data_percentage = ((empty_before - empty_after) / (rows_before * 7)) * 100 if empty_before > 0 else 0
        deleted_data_percentage = ((rows_before - rows_after) / rows_before) * 100 if rows_before > 0 else 0

        logger.info(f"Usunięto {(rows_before - rows_after)} wierszy podczas czyszczenia danych.")
        logger.info(f"Uzupełniono {(empty_before - empty_after)} rekordów podczas wypełniania brakujących danych.")

        df_standardized = standardize_data(df_cleaned)

        # update_google_sheet(df_standardized)

        # # Zapisz raport do pliku
        with open("report.txt", "w") as f:
            f.write(f"Percentage of data changed: {changed_data_percentage:.2f}%\n")
            f.write(f"Percentage of data deleted: {deleted_data_percentage:.2f}%\n")

        return df_standardized

    # Jeśli nie podano żadnej opcji
    else:
        logger.error("Nie podano żadnych opcji. Użyj --upload lub --standardize.")


if __name__ == '__main__':
    main()
