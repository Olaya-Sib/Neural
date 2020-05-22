import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def WISDM_formatting(csv_name):
    type_file = open(csv_name)
    lines = type_file.readlines()

    Processed_List = []
    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                break
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            Processed_List.append(temp)
        except:
            print("Error at line number: ", i)

    columns = ['user', 'activity', 'time', 'x', 'y', 'z']

    edited_database = pd.DataFrame(data=Processed_List, columns=columns)
    edited_database = edited_database.drop(['user', 'time'], axis=1).copy()

    return edited_database


def balance_scale_index(database):
    # defining scaling used
    scale = StandardScaler()

    # making 'x', 'y', 'z' columns float values
    database['x'] = database['x'].astype('float')
    database['y'] = database['y'].astype('float')
    database['z'] = database['z'].astype('float')

    if 'activity' in database.columns:
        # determining number of samples of activity with least samples
        # if database has 'activity' column
        min_value = min(database['activity'].value_counts())

        # taking min_value number of rows from each activity
        Walking = database[database['activity'] == 'Walking'].head(min_value).copy()
        Downstairs = database[database['activity'] == 'Downstairs'].head(min_value).copy()
        Upstairs = database[database['activity'] == 'Upstairs'].head(min_value).copy()
        Standing = database[database['activity'] == 'Standing'].head(min_value).copy()

        # append balanced data
        database = pd.DataFrame()
        database = database.append([Walking, Downstairs, Upstairs, Standing])

        # scale 'x', 'y', 'z' columns from balanced data and create pd.Dataframe()
        x = database[['x', 'y', 'z']]
        x = scale.fit_transform(x)
        scaled_data = pd.DataFrame(data=x, columns=['x', 'y', 'z'])

        # add activity column
        y = database['activity']
        scaled_data['activity'] = y.values

    else:
        # scale 'x', 'y', 'z' columns and create pd.Dataframe()
        x = database[['x', 'y', 'z']]
        x = scale.fit_transform(x)
        scaled_data = pd.DataFrame(data=x, columns=['x', 'y', 'z'])

    scaled_data.index = np.arange(0, len(scaled_data))

    return scaled_data


if __name__ == "__main__":
    WISDM = WISDM_formatting('WISDM_ar_v1.1_raw.txt')
    WISDM = balance_scale_index(WISDM)
    WISDM = WISDM.iloc[::2]

    Validation = pd.read_csv('edited_validation_database.csv')
    Validation = balance_scale_index(Validation)

    Test = pd.read_csv('edited_test_database.csv')
    Test = balance_scale_index(Test)

    WISDM.to_csv("scaled_train_database.csv")
    Validation.to_csv("scaled_validation_database.csv")
    Test.to_csv("scaled_test_database.csv")
