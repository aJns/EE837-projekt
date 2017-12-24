from keras.models import Model
from keras.layers import (Input, Dense, Flatten, Conv2D,
        MaxPooling2D, Dropout, Activation, Reshape) 
from keras import optimizers
from keras.applications import vgg16


def create_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs

## CONV_1 ####################################################################
    filter_count = 128
    x = Conv2D(filters=filter_count, kernel_size=(7,7), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)


## CONV_2 ####################################################################
    filter_count = 256
    x = Conv2D(filters=filter_count, kernel_size=(5,5), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)


## CONV_3 ####################################################################
    filter_count = 512
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)


## CONV_4 ####################################################################
    filter_count = 1024
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)


## CONV_5 ####################################################################
    filter_count = 2048
    x = Conv2D(filters=filter_count, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)


## DENSE_1 ####################################################################
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model




def create_detector(classifier):
    return Model()


if __name__ == "__main__":
    model = create_classifier((80,80,3))
    model.summary()
