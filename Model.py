from Preprocessing import *
from tensorflow import keras
from keras.preprocessing.text import Tokenizer, one_hot
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Model():
 
    def __init__(self, model = None, prep = None):    
        self.model = model
        self.prep = prep
    
    def make_model(self, vocab_size, maxlen):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_dim = vocab_size, output_dim = 50, input_length = maxlen + 1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(50, return_sequences = False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(keras.activations.tanh))
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(keras.activations.relu))
        model.add(keras.layers.Dense(2, activation = 'softmax'))
        model.compile('adam', 'binary_crossentropy', ['accuracy'])
        self.model = model
    
    def fit(self, X, y, validation_data = None, epochs = 5, batch_size = 32):
        if self.prep == None:
            self.prep = preprocess()
        X = self.prep.fit_transform(X)
        y = keras.utils.to_categorical(y)
        
        if validation_data != None:
            validation_data  = list(validation_data)
            validation_data[0] = self.prep.transform(validation_data[0])
            validation_data[1] = keras.utils.to_categorical(validation_data[1])
            validation_data = tuple(validation_data)
        
        if self.model == None: 
            self.make_model(self.prep.vocab_size, self.prep.maxlen)

        cb1 = ReduceLROnPlateau(patience = 3, min_lr = 0.00005, verbose = 1)
        cb2 = EarlyStopping(patience=4, restore_best_weights=True, verbose = 1)
            
        self.model.fit(X, y, epochs = epochs, batch_size = batch_size, validation_data = validation_data, callbacks = [cb1, cb2])
    
    
    def predict(self, X):
        X_test = self.prep.transform(X)
        pred = self.model.predict(X_test)
        return pred