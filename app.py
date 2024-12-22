import pandas as pd
import joblib
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer  

import os
import config

# preprocessing pipeline
def preprocessing_pipeline():
    # select columns
    num_cols = make_column_selector(dtype_include='number')
    cat_cols = make_column_selector(dtype_include='object')

    # Instantiate transformer
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore')
    imputer = KNNImputer(n_neighbors=2, weights='uniform') # Use KNN algorithm to impute missing values to preserve data integrity
    
    # Create pipeline
    num_pipe = Pipeline([
        ('scaler', scaler),
        ('imputer', imputer)
    ])
    
    cat_pipe = Pipeline([
        ('encoder', encoder)
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols)
    ])
    
    return preprocessor

# Encode categorical variables
def categorical_encoder(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes
        
    return df

@st.cache_resource
def load_models():
    lgb_model = joblib.load(config.MODEL_LGBM_PATH)
    dec_tree = joblib.load(config.MODEL_DEC_TREE_PATH)
    
    return lgb_model, dec_tree

@st.cache_data
# function to load and fit the preprocessor
def load_and_fit_preprocessor(file_path):
    flight_df = pd.read_csv(file_path)
    preprocessor = preprocessing_pipeline()
    preprocessor.fit(flight_df)
    
    return preprocessor, flight_df

# function to collect user inputs
def collect_user_inputs():
    # collect user inputs
    inputs = {
        # feature: st.sidebar.slider(app label, min, max, default)
        'Online boarding': st.sidebar.slider('Online Boarding', 1, 5, 3),
        'Inflight wifi service': st.sidebar.slider('Inflight WIFI Service', 1, 5, 3),
        'Inflight entertainment': st.sidebar.slider('Inflight Entertainment', 1, 5, 3),
        'Checkin service': st.sidebar.slider('Check in Service', 1, 5, 3),
        'Seat comfort': st.sidebar.slider('Seat Comfort', 1, 5, 3),
        'Age': st.sidebar.number_input('Age', 7, 100, 18),
        'Flight Distance': st.sidebar.number_input('Flight Distance', 31, 10000, 100),
        'Business travel': st.sidebar.selectbox('Business Travel', ['Yes', 'No']),
        'Loyal Customer': st.sidebar.selectbox('Loyal Customer', ['Yes', 'No']),
        'Class': st.sidebar.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
    }
    
    return pd.DataFrame(inputs, index=[0])
    
# make predictions and explain the results
def make_predictions(model, preprocessor, input_df, lime_explainer):
    
    try:
        # Transform user inputs using the preprocessor
        input_df_transformed = preprocessor.transform(categorical_encoder(input_df))
        prediction = model.predict(input_df_transformed)
        
        # LIME explanation
        lime_explanation = lime_explainer.explain_instance(
            data_row=input_df_transformed[0],
            predict_fn=model.predict_proba,
            num_features = config.LIME_NUM_FEATURES
        )
        
        return 'Satisfied' if prediction[0] else 'Not Satisfied', input_df_transformed, lime_explanation
        
        
    except Exception as error:
        st.error(f'An error occurred: {str(error)}. Please contact customer service.')
        
        return None, None, None

# Display logo and hero image
st.sidebar.image('images/logo.png', width=100)
st.image('images/hero.jpg', use_container_width=True)

st.title('Flight Satisfaction Prediction App')
st.header('Please Tell Us About Your Experience')

# load models and preprocessor
lgb_model, dec_tree = load_models()
preprocessor, flight_df = load_and_fit_preprocessor(config.DATA_PATH)

# model selection
model_choice = st.sidebar.selectbox('Select Model', ['LightGBM', 'Decision Tree'])
selected_model = lgb_model if model_choice == 'LightGBM' else dec_tree

# initialize LIME Explainer
lime_explainer = LimeTabularExplainer(
    training_data = preprocessor.transform(categorical_encoder(flight_df)),
    feature_names = flight_df.columns,
    mode='classification'
)

# add app description

# collect user inputs
