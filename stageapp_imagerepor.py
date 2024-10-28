import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # Ensure joblib is imported
from sklearn.preprocessing import MinMaxScaler

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="stage Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.image("cancer.jpg", width=800)  # Change "logo.png" to the path of your logo image file
# Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">MultiModel Cancer Stage Prediction</h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #800080;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

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
    'Education_Level': 'categorical',
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
    'Is Available': 'numerical',
}

# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example XLSX input file](https://master/penguins_example.csv)
""")

# Create dictionary for grouping labels
group_labels = {
    'Demographic Data': ['Age', 'Sex', 'Occupation', 'Education_Level', 'Residence', 'Region', 'Zone', 'City', 'SubCity', 'Woreda', 'Kebel'],
    'Clinical Data': ['Diagnosis', 'Group diagonsis', 'Type diagnosis',
       'Status', 'Unit', 'Pacient Weight', 'BMI', 'Laboratory Service ', 'HistoryType', 'History value ',
       'Prescrption Type', 'Prescribed Item',  'Tumor_Type'],
    'Imagereport Data': ['Imagereport'],
    'Financial Data': ['Price', 'Is Paid', 'Is Available'],
}

# Option for CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])

# If CSV file is uploaded, read the file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# If CSV file is not uploaded, allow manual input
else:
    # Create empty dataframe to store input values
    input_df = pd.DataFrame(index=[0])

    # Loop through features and get user input
    for group, features_in_group in group_labels.items():
        st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
        st.sidebar.subheader(group)
        for feature in features_in_group:
            # Ensure each widget has a unique key
            widget_key = f"{group}_{feature}"

            # Display more descriptive labels
            if features[feature] == 'categorical':
                label = f"{feature.replace('_', ' ')}"
                input_df[feature] = st.sidebar.selectbox(label, label_encoders[feature].classes_, key=widget_key)
            else:
                label = f"{feature.replace('_', ' ')}"
                input_val = st.sidebar.text_input(label, key=widget_key)
                input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Additional styling for the overview section
st.markdown(
    """
    ### Welcome to Cancer Stage prediction/ Tool!

    #### What You Can Do:
   1. Check cancer stages .i.e. which cancer stages have the pacients.
   2. To make decisions for patients and physicians that have cancer pacients stage accordingly
   3. make it easy to follow up with the patients 
   4. which catagory of the pacients have need spacial treatments most likely severl stage such as Stage4 and Stage3 

    Dive into the rich data of Tikur Anbessa Hosipitals from 2020 to 2024, interact, and uncover valuable insights for decision making!
    """
)

# Display the input dataframe
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'Stage' and feature in input_df.columns:
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-Max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        if feature in input_df.columns:  # Check if feature exists in input_df
            try:
                # Perform scaling
                input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))
            except ValueError as e:
                st.sidebar.write(f"Error scaling {feature}: {e}")
                # Optionally set to a default value if needed
                # input_df[feature] = np.nan  # or any default value you choose

    # Display the input data after encoding and normalization
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)

    # Prepare data for prediction
    model_features = loaded_model.get_booster().feature_names
    input_df_filtered = input_df[model_features]  # Only select model features

    # Make predictions
    prediction = loaded_model.predict(input_df_filtered)

    # Ensure prediction is a valid array with expected shape
    if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
        # Assuming it's a classification problem, take the first prediction
        predicted_label = prediction[0]  # Get the first prediction

        # Output the prediction
        Stage = np.array(['stage1', 'stage2', 'stage3', 'stage4'])  # Ensure correct labels based on your model
        prediction_index = int(predicted_label)  # Make sure this is correctly indexed

        # Output the prediction
        st.sidebar.write("Prediction:", Stage[prediction_index])

        # Show prediction probabilities if applicable
        if hasattr(loaded_model, 'predict_proba'):
            prediction_proba = loaded_model.predict_proba(input_df_filtered)
            st.subheader('Prediction (which stage is the Cancer?)')
            st.write(f"stage cancer: {Stage[prediction_index]}")

            st.subheader('Prediction Probability')
            probability_df = pd.DataFrame(prediction_proba, columns=Stage)
            st.write(probability_df)
    else:
        st.sidebar.write("Prediction could not be made.")
