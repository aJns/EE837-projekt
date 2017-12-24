import os

from model import create_classifier, create_detector
from data import load_data


# Define constants
OUTPUT_PATH = "/output"
DATA_PATH = "/data/shipsnet"


## Load data
(x_train, y_train) = load_data(DATA_PATH)
data_shape = x_train.shape[1:]


## Train classifier
classifier = create_classifier(data_shape)
classifier.fit(x_train, y_train, batch_size=100, epochs=20, validation_split=0.1, shuffle=True)


## Save output
if os.path.exists(OUTPUT_PATH):
    classifier.save(OUTPUT_PATH + "/trained_model.h5")



## Evaluate
