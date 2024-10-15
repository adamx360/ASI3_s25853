import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


def log_action(message):
    logging.info(message)
    print(message)


# Przykład
log_action("Rozpoczęcie czyszczenia danych")

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
client = gspread.authorize(creds)

sheet = client.open("spreadsheet").sheet1
data = sheet.get_all_records()


def clean_and_standardize(data):
    # Uzupełnianie braków np. medianą
    data.fillna(data.median(), inplace=True)

    # Usuwanie wierszy z nadmiarem braków
    data.dropna(thresh=2, inplace=True)

    # Standaryzacja
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[float, int]))

    # Powrót do DataFrame
    data.loc[:, data.select_dtypes(include=[float, int]).columns] = scaled_data
    return data


def generate_report(data, original_data):
    changed = (data != original_data).sum().sum()
    removed = len(original_data) - len(data)

    report = f"Zmodyfikowane dane: {changed}\nUsunięte dane: {removed}\n"

    with open("report.txt", "w") as f:
        f.write(report)


def main():
    print("sa")


if __name__ == '__main__':
    main()
