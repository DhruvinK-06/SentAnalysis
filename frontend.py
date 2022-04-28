import streamlit as st
from Model import * 
import os
from tensorflow import keras
import pickle

@st.cache(allow_output_mutation=True)
def load_files():
    path = os.path.dirname(__file__)
    model_path = os.path.join(path, 'Saved Files', 'Model.h5')
    prep_path = os.path.join(path, 'Saved Files', 'Preprocessing')

    mod = keras.models.load_model(model_path)
    file = open(prep_path, 'rb')
    prep = pickle.load(file)

    return mod, prep

mod, prep = load_files()

model = Model(mod, prep)

header = st.container()
Mod = st.container()

with header:
    st.title('😡😀 Sentiment Analysis 😀😡')
    st.write('\n')
    st.write('\n')
    st.write('\n')

with Mod:
    txt = st.text_input('Enter the review to do the analysis on:', '')
    pred = np.argmax(model.predict([txt]))
    col = st.columns(5)
    col1 = st.columns(3)
    st.write('\n')
    if col[2].button('Enter'):
        if txt != '':
            if pred == 0:
                col1[1].write('😡 The entered review is negative 😡')
            else:
                col1[1].write('😀 The entered review is positive 😀')
        else:
            col1[1].write('😐 Please enter a review first!! 😐')




    


