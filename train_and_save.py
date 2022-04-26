from Model import *
import pandas as pd
import os
import pickle

path = os.path.dirname(__file__)
data_path = os.path.join(path, 'Data')
train_path = os.path.join(data_path, 'Train.csv')
valid_path = os.path.join(data_path, 'Valid.csv')
df = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)

model = Model()
model.fit(df['text'], df['label'], validation_data = (valid['text'], valid['label']), epochs = 12, batch_size = 64)

prep_path = os.path.join(path, 'Saved Files', 'Preprocessing')
prep_file = open(prep_path, 'wb')
pickle.dump(model.prep, prep_file)

model_path = os.path.join(path, 'Saved Files', 'Model.h5')
model.model.save(model_path)
