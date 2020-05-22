# KGP Talkie: https://www.youtube.com/watch?v=lUI6VMj43PE

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from FunctionDefinitions import create_segments_and_labels, create_segments
from sklearn.preprocessing import LabelEncoder

# defining label encoder
label = LabelEncoder()

# reading CSVs to pandas dataframe objects
train_database = pd.read_csv("scaled_train_database.csv")
validation_database = pd.read_csv("scaled_validation_database.csv")
test_database = pd.read_csv("scaled_test_database.csv")


# encoding databases that have 'activity' columns
def encode_databases(database1, database2):
    database1['label'] = label.fit_transform(database1['activity'].values.ravel())
    database2['label'] = label.transform(database2['activity'].values.ravel())
    return database1, database2


train_database, validation_database = encode_databases(train_database, validation_database)

Fs = 10  # in video this values was 20 (because of 20Hz)
frame_size = 9  # in video this value was Fs*4 (80)
hop_size = 10  # in video this value was Fs*2

x_train, y_train = create_segments_and_labels(df=train_database,
                                              time_steps=frame_size,
                                              step=hop_size,
                                              label_name='label'
                                              )

x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y_train
                                                    )

x_val, y_val = create_segments_and_labels(df=validation_database,
                                          time_steps=frame_size,
                                          step=hop_size,
                                          label_name='label'
                                          )


# reshaping: since model accepts 4D data, x_train x_test and x_val must be reshaped as follows:
x_train = x_train.reshape(568, 9, 3, 1)
x_test = x_test.reshape(143, 9, 3, 1)
x_val = x_val.reshape(105, 9, 3, 1)

# layering model
model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation="softmax"))

# compiling model
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# fitting model
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), verbose=1)

# evaluating model and print out performance
scores = model.evaluate(x_val, y_val, verbose=1)
print("val_loss:", scores[0], "accuracy:", scores[1])

# making test data
x_test = create_segments(df=test_database,
                         time_steps=frame_size,
                         step=hop_size
                         )

# re-shaping test data
x_test = x_test.reshape(125, 9, 3, 1)
program_encoded_class_results = model.predict_classes(x_test, verbose=1)
label_class_result = label.inverse_transform(program_encoded_class_results)

# results to be submitted
predicted_labels_coded = []
for i in range(len(label_class_result)):
    if label_class_result[i] == 'Standing':
        predicted_labels_coded.append(1)
    elif label_class_result[i] == 'Walking':
        predicted_labels_coded.append(2)
    elif label_class_result[i] == 'Downstairs':
        predicted_labels_coded.append(3)
    elif label_class_result[i] == 'Upstairs':
        predicted_labels_coded.append(4)

print(predicted_labels_coded)
print(len(predicted_labels_coded))


# we have our model from KGP Talkie tested on the train time_series/ train_labels given from course
# performs ~ 50% accuracy & trained with WISDM dataset
