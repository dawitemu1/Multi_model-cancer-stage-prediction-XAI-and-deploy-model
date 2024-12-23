import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # Ensure joblib is imported
from sklearn.preprocessing import MinMaxScaler

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="Stage Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS for styling
st.markdown("""
<style>
    div.stApp { margin-top: -90px !important; }
    body { background-color: #f5f5f5; }
    h1 { color: #800080; font-family: 'Helvetica', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.image("cancer.jpg", width=800)
st.markdown('<h1 style="text-align: center;">MultiModel Cancer Stage Prediction</h1>', unsafe_allow_html=True)

# Load the model, label encoders, and scalers
# Load the CatBoost model
loaded_model = joblib.load('xgb_model2.pkl')

# Load the label encoders
label_encoders = joblib.load('label_encoders2.pkl')

# Load the MinMax scaler parameters
minmax_scalers = joblib.load('scaler_params2.pkl')
# Feature names and types
features = {
    'Age': 'numerical',
    'Sex': 'categorical', 
    'Occupation': 'categorical',
    'Education_Level':   'categorical', 
    'Residence': 'categorical', 
    'Region': 'categorical',
    'Zone': 'categorical', 
    'City': 'categorical', 
    'SubCity': 'categorical',
    'Woreda': 'categorical',
    'Kebel': 'categorical', 
    'Diagnosis': 'categorical',
    'Group diagonsis': 'categorical', 
    'Type diagnosis': 'categorical', 
    'Status': 'categorical',
    'Unit': 'categorical', 
    'Pacient Weight': 'categorical', 
    'BMI': 'categorical',
    'Laboratory Service ': 'categorical', 
    'HistoryType': 'categorical', 
    'History value ': 'categorical',
    'Prescrption Type': 'categorical',
    'Prescribed Item': 'categorical',
    'Tumor_Type': 'categorical',
    'Imagereport': 'categorical',
    'Price': 'numerical', 
    'Is Paid': 'numerical', 
    'Is Available': 'numerical'
}
# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example XLSX input file](https://master/penguins_example.csv)
""")

# Group labels
group_labels = {
    'Demographic Data': ['Age', 'Sex', 'Occupation', 'Education_Level', 'Residence', 'Region', 'Zone', 'City', 'SubCity', 'Woreda', 'Kebel'],
    'Clinical Data': ['Diagnosis', 'Group diagonsis', 'Type diagnosis', 'Status', 'Unit', 'Pacient Weight', 'BMI', 'Laboratory Service ', 'HistoryType', 'History value ', 'Prescrption Type', 'Prescribed Item', 'Tumor_Type'],
    'Imagereport Data': ['Imagereport'],
    'Financial Data': ['Price', 'Is Paid', 'Is Available']
  }

# Option for XLSX file upload
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])
# Initialize input dataframe
input_df = pd.DataFrame(index=[0])

# Define columns for Demographic, Clinical, Financial, and Image data
demographic_col, clinical_col, financial_col, image_col = st.columns(4)

# Demographic Data Inputs
with demographic_col:
    st.subheader("Demographic Data")
    for feature in group_labels['Demographic Data']:
        widget_key = f"Demographic_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Clinical Data Inputs
with clinical_col:
    st.subheader("Clinical Data")
    for feature in group_labels['Clinical Data']:
        widget_key = f"Clinical_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Financial Data Inputs
with financial_col:
    st.subheader("Financial Data")
    for feature in group_labels['Financial Data']:
        widget_key = f"Financial_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Image Data Inputs
# Image Data Inputs
with image_col:
    st.subheader("Image Data")
    for feature in group_labels['Imagereport Data']:  # Corrected from 'Image' to 'Imagereport Data'
        widget_key = f"Image_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')


# Display the input data before encoding and normalization
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    # Encode categorical features and scale numerical features
    for feature, encoder in label_encoders.items():
        if feature != 'Stage' and feature in input_df.columns:
            input_df[feature] = encoder.transform(input_df[feature])

    for feature, scaler in minmax_scalers.items():
        if feature in input_df.columns:
            input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))

    # Display the processed input data
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)

    # Prepare data for prediction
    model_features = loaded_model.get_booster().feature_names
    input_df_filtered = input_df[model_features]

    # Make predictions
    prediction = loaded_model.predict(input_df_filtered)

    if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
        predicted_label = prediction[0]  # Assuming first prediction
        Stage = np.array(['stage1', 'stage2', 'stage3', 'stage4'])
        prediction_index = int(predicted_label)

        # Display prediction
        st.sidebar.write("Prediction:", Stage[prediction_index])
        st.subheader('Prediction (In which Stage is Cancer?)')
        st.write(f"Stage: {Stage[prediction_index]}")

        if hasattr(loaded_model, 'predict_proba'):
            prediction_proba = loaded_model.predict_proba(input_df_filtered)
            st.subheader('Prediction Probability')
            probability_df = pd.DataFrame(prediction_proba, columns=Stage)
            st.write(probability_df)
    else:
        st.sidebar.write("Prediction could not be made.")
