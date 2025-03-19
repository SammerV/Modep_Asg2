#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np

# Load the model and scaler from the pickle file
with open('SVM.pickle', 'rb') as f:
    saved_data = pickle.load(f)
    loaded_model = saved_data['model']
    loaded_scaler = saved_data['scaler']

def main():
    st.title('Iris Flower Species Prediction')
    
    # Input sliders for features
    sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.slider('Petal Length', min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.slider('Petal Width', min_value=0.0, max_value=10.0, value=0.2)
    
    # Make prediction button
    if st.button('Make Prediction'):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        result = make_prediction(features)
        st.success(f'The predicted species is: {result}')

def make_prediction(features):
    # Convert input to numpy array and reshape
    input_array = np.array(features).reshape(1, -1)
    
    # Preprocess the input using the loaded scaler
    input_array = loaded_scaler.transform(input_array)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_array)
    
    # Map prediction to species name
    species = ['setosa', 'versicolor', 'virginica']
    return species[prediction[0]]

if __name__ == '__main__':
    main()


# In[ ]:




