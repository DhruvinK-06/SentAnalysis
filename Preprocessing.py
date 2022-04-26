import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
class preprocess():
    def __init__(self):
        pass
    
    def process(self, X, count = False):
        swords = stopwords.words('english')
        unique_words = set()
        stemmer = PorterStemmer()
        df = X.copy()
        for i in range(len(df)):
            txt = df[i]
            txt = txt.lower()
            txt = re.sub(r'[^\w\s]', '', txt)
            final = []
            words = txt.split(' ')
            for word in words:
                if word not in swords:
                    w = stemmer.stem(word)
                    final.append(w)
                    if count:
                        unique_words.add(w)
            df[i] = ' '.join(final)
        return df, len(unique_words)
    
    def fit(self, X, y = None):
        X_, self.vocab_size = self.process(X, True)
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\]^_`{|}~')
        self.tokenizer.fit_on_texts(X_)
        self.sequences = self.tokenizer.texts_to_sequences(X_)
        self.maxlen = max([len(sequence) for sequence in self.sequences])
        return self
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        X_ = pad_sequences(self.sequences, maxlen = self.maxlen + 1, padding='pre')
        return X_

    def transform(self, X, y = None):
        X_, vs = self.process(X)
        seq = self.tokenizer.texts_to_sequences(X_)
        X_ = pad_sequences(seq, maxlen = self.maxlen + 1, padding='pre')
        return X_
        