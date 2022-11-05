from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model


def final_model(input_data, kernel_size, dropout, padding, n_classes):
    """
    Creates convolutional neural network
    Args:
        input_data (tuple): Size of our data
        kernel_size (tuple): Size of kernel
        dropout (list): Value of dropout SIZE = 2!!!
        padding (string): Prevent data from information loss
        n_classes (int): number of classes
    Return:
        Model
    """
    input_layer = Input(shape=input_data)

    x = Conv1D(filters=128,
               kernel_size=kernel_size - 1,
               padding=padding,
               activation='relu')(input_layer)
    x = MaxPooling1D()(x)

    x = Conv1D(filters=128,
               kernel_size=kernel_size,
               padding=padding,
               activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Dropout(dropout[0])(x)

    x = Conv1D(filters=256,
               kernel_size=kernel_size + 1,
               padding=padding,
               activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Dropout(dropout[0])(x)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout[1])(x)
    x = BatchNormalization()(x)

    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)
    return model


