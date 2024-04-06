import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load transformers and model from pickle file
with open('transformers_and_model.pickle', 'rb') as handle:
    transformers_and_model_loaded = pickle.load(handle)

#access the loaded objects
label_encoder_car = transformers_and_model_loaded['label_encoder_car']
label_encoder_location = transformers_and_model_loaded['label_encoder_location']
scaler_loaded = transformers_and_model_loaded['scaler']
selector_loaded = transformers_and_model_loaded['feature_selector']
log_reg_loaded = transformers_and_model_loaded['logistic_regression']

#function to preprocess input data
def preprocess_data(data):
    #create a DataFrame with all features
    all_features = ['Staff', 'Floor Space', 'Window', 'Car park', 
                    'Demographic score', 'Location', '40min population',
                    '30 min population', '20 min population', '10 min population',
                    'Store age', 'Clearance space', 'Competition score']
    all_data = pd.DataFrame(columns=all_features)

    #fill in selected features
    selected_features = ['Staff', 'Window', 'Location', 'Competition score']
    for feature in selected_features:
        all_data[feature] = data[feature]

    #fill in default values or drop missing features
    for feature in all_features:
        if feature not in selected_features:
            #if the feature was not selected, fill in default value or drop it
            if feature.startswith('pop'):
                #fill in 0 for population features
                all_data[feature] = 0
            elif feature == 'Car park':
                #fill in default value for Car park
                all_data[feature] = 'No'
            else:
                #drop other features
                all_data.drop(columns=[feature], inplace=True)

    if 'Car park' in all_data.columns:
        all_data['Car park'] = label_encoder_car.transform(all_data['Car park'])
    #apply transformations
    for feature in all_features:
        if feature not in all_data.columns:
            all_data[feature] = 0 
    all_data = all_data.reindex(columns=all_features)
    data_scaled = scaler_loaded.transform(all_data)
    data_selected = selector_loaded.transform(data_scaled)
    return data_selected


#function to predict using the loaded model
def predict(data):
    data_preprocessed = preprocess_data(data)
    predictions = log_reg_loaded.predict(data_preprocessed)
    return predictions

#function to create web page
def main():
    st.title('Store Performance Prediction')

    #create input fields for selected features
    staff = st.slider('Staff', min_value=0, max_value=10, value=10, step=1)
    window = st.slider('Window', min_value=0, max_value=130, value=5, step=1)
    location = st.selectbox('Location', label_encoder_location.classes_)
    competition_score = st.slider('Competition Score', min_value=0, max_value=20, value=10, step=1)
   
   
    #map location to numerical value
    location_encoded = label_encoder_location.transform([location])[0]
    ok = st.button("Show Performance Prediction")
    #create a DataFrame with input data
    if ok:
        input_data = pd.DataFrame({
        'Staff': [staff],
        'Window': [window],
        'Location': [location_encoded],
        'Competition score': [competition_score]
        })

        #make prediction
        prediction = predict(input_data)

        #display prediction result
        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.write('The store is predicted to have good performance 😃 ')
        else:
            st.write('The store is predicted to have poor performance ☹️ ')

if __name__ == '__main__':
    main()
