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
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

# -------------------------------
# 1) Cancer stage colors
# -------------------------------
stage_colors = {
    'stage1': 'green',
    'stage2': 'orange',
    'stage3': 'blue',
    'stage4': 'red'
}

# -------------------------------
# 2) Load shapefile
# -------------------------------
shp_path = r"C:\Users\Daveee\Downloads\Regions-old\Regions-old\Regions-old.shp"
ethiopia = gpd.read_file(shp_path)
ethiopia = ethiopia[ethiopia.is_valid & ~ethiopia.is_empty]

# -------------------------------
# 3) Build region polygon mapping
# -------------------------------
region_polygons = {row['NAME_1']: row['geometry'] for _, row in ethiopia.iterrows()}

# -------------------------------
# 4) Function to generate random point inside polygon
# -------------------------------
def random_point_within(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            return x, y

# -------------------------------
# 5) Interactive plot with very small scatter
# -------------------------------
# For Jupyter notebook: enable interactive backend
# %matplotlib notebook  # Uncomment if using Jupyter

fig, ax = plt.subplots(figsize=(12, 12))
ethiopia.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=0.5)

for idx, row in df.iterrows():
    region = row['Region']
    stage = row['Stage']

    if region not in region_polygons:
        continue

    poly = region_polygons[region]
    lon, lat = random_point_within(poly)

    ax.scatter(
        lon, lat,
        color=stage_colors.get(stage, 'black'),
        s=1,       # VERY SMALL point
        alpha=0.3,
        zorder=5
    )

# -------------------------------
# 6) Legend and labels
# -------------------------------
stage_handles = [
    plt.Line2D([], [], marker='o', color=color, label=stage, markersize=6)
    for stage, color in stage_colors.items()
]
plt.legend(handles=stage_handles, title='Cancer Stage', fontsize=10)
plt.title("Cancer Stage Distribution Across Ethiopia (Interactive Zoomable)", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Optional: region labels at centroid
for region, poly in region_polygons.items():
    centroid = poly.centroid
    ax.text(centroid.x, centroid.y, region, fontsize=6, ha='center', va='center', alpha=0.7)

plt.tight_layout()

# Enable interactive navigation (zoom/pan)
plt.show()






