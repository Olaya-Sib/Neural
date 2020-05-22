# this code is in python file named Editing_Course_CSVs.py

import pandas as pd


def merge_processing(database):
    del database['Unnamed: 0_y']
    del database['UTC time_x']
    del database['UTC time_y']
    del database['accuracy']
    del database['timestamp']
    del database['Unnamed: 0_x']
    database.columns = ['x', 'y', 'z', 'activity']
    return database


def activity_decoding(database):

    def label_activity(row):
        if row['activity'] == 1:
            return 'Standing'
        elif row['activity'] == 2:
            return 'Walking'
        elif row['activity'] == 3:
            return 'Downstairs'
        elif row['activity'] == 4:
            return 'Upstairs'

    database['activity'] = database.apply(lambda row: label_activity(row), axis=1)

    return database


if __name__ == "__main__":
    # Merged train CSVs (i.e., train time-series + train labels)
    train_labels = pd.read_csv('train_labels.csv')
    train_time_series = pd.read_csv('train_time_series.csv')
    merged_database = pd.merge(left=train_time_series, right=train_labels, how='left', left_on="timestamp", right_on="timestamp")
    merged_database = merge_processing(merged_database)
    # since label is only given every 10th observation,
    # I used fillna to turn nan values to correct activities
    merged_database['activity'].fillna(method='backfill', inplace=True)
    merged_database = activity_decoding(merged_database)

    # Edited test CSV
    test_time_series = pd.read_csv('test_time_series.csv')
    del test_time_series['UTC time']
    del test_time_series['accuracy']
    del test_time_series['Unnamed: 0']
    del test_time_series['timestamp']
    test_time_series.columns = ['x', 'y', 'z']

    # Saved data-frame objects as edited CSVs
    merged_database.to_csv('edited_validation_database.csv')
    test_time_series.to_csv('edited_test_database.csv')

