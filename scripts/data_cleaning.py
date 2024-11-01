import os
import logging
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder


def setup_logger():
    logger = logging.getLogger('DataCleaningLogger')
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('../log.txt', mode='w')
    c_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


def main():
    logger = setup_logger()
    logger.info('Starting data cleaning process.')

    data_file_path = os.path.join('../data', 'CollegeDistance.csv')

    try:
        df = pd.read_csv(data_file_path)
        logger.info('Data loaded successfully.')
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        sys.exit(1)

    logger.info(f"First few rows of the data:\n{df.head()}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    total_cells = df.size
    missing_values_before = df.isnull().sum().sum()
    rows_before = df.shape[0]

    df = df[df.isnull().sum(axis=1) <= 1]

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    missing_values_after = df.isnull().sum().sum()
    filled_values = missing_values_before - missing_values_after

    rows_after = df.shape[0]
    rows_removed = rows_before - rows_after

    logger.info('Performing feature engineering...')

    distance_bins = [0, 1, 5, df['distance'].max()]
    distance_labels = ['Near', 'Medium', 'Far']
    df['distance_category'] = pd.cut(df['distance'], bins=distance_bins, labels=distance_labels)

    df['parents_college'] = df.apply(lambda row: 'both' if row['fcollege'] == 'yes' and row['mcollege'] == 'yes'
    else ('father' if row['fcollege'] == 'yes' else ('mother' if row['mcollege'] == 'yes' else 'none')), axis=1)

    df['unemp_wage_interaction'] = df['unemp'] * df['wage']

    new_categorical_features = ['distance_category', 'parents_college']
    df = pd.get_dummies(df, columns=new_categorical_features, drop_first=True)

    logger.info('Feature engineering completed.')

    if 'score' in df.columns:
        df['score'] = df['score'].round(2)
        logger.info("Rounded 'score' to two decimal places.")

    logger.info('Generating report...')
    try:
        percent_filled = (filled_values / total_cells) * 100
        percent_removed = (rows_removed / rows_before) * 100
        with open('../report.txt', 'w') as f:
            f.write(f'Percentage of filled missing data: {percent_filled:.2f}%\n')
            f.write(f'Percentage of removed rows: {percent_removed:.2f}%\n')
            f.write("Data has been cleaned, features engineered, and 'score' rounded to two decimal places.\n")
            f.write("Rows with more than one missing field have been removed.\n")
        logger.info('Report generated.')
    except Exception as e:
        logger.error(f'Error generating report: {e}')
        sys.exit(1)

    logger.info(f'Filled {filled_values} missing values ({percent_filled:.2f}% of total data).')
    logger.info(f'Removed {rows_removed} rows ({percent_removed:.2f}% of total rows).')

    cleaned_data_path = '../cleaned_data.csv'
    df.to_csv(cleaned_data_path, index=False)
    logger.info(f'Cleaned data saved to {cleaned_data_path}.')
    logger.info('Data cleaning process completed.')


if __name__ == '__main__':
    main()
