import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i: i + window_size] for i in range(len(series) - window_size)]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    # Layer 1: LSTM
    model.add(LSTM(5, input_shape=(window_size, 1)))

    # Layer 2: Fully connected layer
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # Get ascii charaters set 
    ascii = string.ascii_lowercase

    # Get typical characters
    typical_chars = ascii + ''.join(punctuation)

    # Remove all atypical characters
    text = ''.join(c if c in typical_chars else ' ' for c in text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    textLength = len(text)

    inputs = [text[i:i + window_size] for i in range(0, textLength - window_size, step_size)]
    
    outputs = [text[i] for i in range(window_size, textLength, step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()

    # Layer 1: LSTM 
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    
    # Layer 2: Fully connected layer
    model.add(Dense(num_chars, activation='linear'))

    # Layer 3: Softmax activation
    model.add(Activation('softmax'))

    return model
