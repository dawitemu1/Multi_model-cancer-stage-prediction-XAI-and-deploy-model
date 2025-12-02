## MultiModel Cancer Stage Prediction Tool

Welcome to Cancer Stage prediction/ Tool!

What You Can Do:

* Check cancer stages .i.e. which cancer stages have the pacients.

* To make decisions for patients and physicians that have cancer pacients stage accordingly

* make it easy to follow up with the patients

* which catagory of the pacients have need spacial treatments most likely severl stage such as Stage4 and Stage3

Dive into the rich data of Tikur Anbessa Hosipitals from 2020 to 2024, interact, and uncover valuable insights for decision making!

We Insatll All required Library such as, Dash, Streamlit, Gradio for front and back end developemnt 
## Save The following

= Save Best perfomed model, XGBoost

= Save Label encoder for Catagorical features 

= MinMax scaler for Numerical feature 
## Demo


https://github.com/user-attachments/assets/4e75e784-fed2-4e4b-a614-5acd424685ac



# Multi-Model Cancer Stage Prediction with XAI and Deployment

## Overview
This repository provides a comprehensive framework for predicting cancer stages using multiple machine learning models, with an emphasis on Explainable AI (XAI) and deployment-ready solutions. The system assists clinicians in accurate staging, improving decision-making and patient follow-up.

The dataset consists of patient records from Tikur Anbessa Hospital (2020–2024).

---

## Features
- **Cancer Stage Prediction**: Predicts the stage of cancer for each patient (stage1, stage2, stage3, stage4).  
- **Explainable AI**: Integrates XAI techniques like SHAP and LIME for model interpretability.  
- **Balanced Dataset Handling**: Applies SMOTE to handle class imbalance.  
- **Multi-Model Support**: Ensemble models including XGBoost, CatBoost, Random Forest, and LightGBM.  
- **Deployment-Ready**: Frontend implemented using Streamlit, Dash, and Gradio.  

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dawitemu1/Multi_model-cancer-stage-prediction-XAI-and-deploy-model.git
   cd Multi_model-cancer-stage-prediction-XAI-and-deploy-model
## Create a virtual environment (recommended):
python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

### Install dependencies:

pip install -r requirements.txt

## Cancer Stage Mapping in Ethiopia

This repository includes functionality to visualize the spatial distribution of cancer stages across Ethiopia.

### Workflow
1. **Shapefile Loading**  
   - Read the Ethiopian administrative regions shapefile using `geopandas`.
2. **Data Integration**  
   - Merge patient records containing cancer stage and region information with the shapefile.
3. **Geolocation of Cancer Stages**  
   - Assign each patient’s stage to the corresponding region.
4. **Visualization**  
   - Generate maps displaying the distribution of cancer stages across regions, helping identify regional patterns and hotspots.

<img width="975" height="794" alt="image" src="https://github.com/user-attachments/assets/4ef643f5-01a1-47e7-b99e-ff2b1d8b8a92" />


### Example (Python)
```python
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load Ethiopian regions shapefile
ethiopia_map = gpd.read_file('data/Ethiopia_regions.shp')

# Load patient data
df = pd.read_csv('data/patient_records.csv')

# Merge shapefile with cancer stage data
map_data = ethiopia_map.merge(df, left_on='Region', right_on='Region', how='left')

# Plot cancer stage distribution
map_data.plot(column='Stage', legend=True, cmap='OrRd', figsize=(10,10))
plt.title('Distribution of Cancer Stages Across Ethiopia')
plt.show()





