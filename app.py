import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import os

# --- Page Config ---
st.set_page_config(page_title="Air Pollution Analysis Results", layout="wide")

# --- High-Contrast UI Design ---
st.markdown("""
    <style>
    .stApp { background-color: #050A18; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #374151; }
    
    /* Content Boxes */
    .result-box {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    
    /* Text Visibility */
    h1, h2, h3 { color: #F3F4F6; }
    .result-box h4 { color: #60A5FA; margin-top: 0; }
    .result-box p, .result-box li { color: #E5E7EB; font-size: 1.05rem; }
    
    /* Dataframe Visibility */
    .stDataFrame { background-color: #FFFFFF; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- Logic from your IPYNB ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'Air_Pollution.csv')

@st.cache_data
def run_analysis_logic():
    if not os.path.exists(FILE_PATH): return None, None
    
    # 1. Loading and Cleaning (Cell 1 & 2)
    df_raw = pd.read_csv(FILE_PATH)
    df = df_raw.dropna(subset=['pollutant_avg', 'latitude', 'longitude']).copy()
    
    # 2. Transforming data for the model (Cell 5 & 11)
    df_p = pd.get_dummies(df, columns=['pollutant_id'], drop_first=True)
    df_p.columns = [c.replace('.', '_') for c in df_p.columns]
    
    # 3. Scaling Latitude/Longitude (Cell 7)
    scaler = StandardScaler()
    df_p[['latitude', 'longitude']] = scaler.fit_transform(df_p[['latitude', 'longitude']])
    
    # 4. Creating Advanced Features (Cell 13)
    df_p['latitude2'] = df_p['latitude']**2
    df_p['longitude2'] = df_p['longitude']**2
    df_p['lat_long_interaction'] = df_p['latitude'] * df_p['longitude']
    
    return df_raw, df_p

df_raw, df_p = run_analysis_logic()

if df_raw is None:
    st.error("Missing Air_Pollution.csv file in the GitHub repository.")
    st.stop()

# --- Dashboard Layout ---
st.title("🌬️ Air Pollution Analysis: Results & Outcomes")
st.markdown("---")

menu = st.sidebar.radio("Analysis Steps", ["Data Overview", "Location & Interaction Results", "Final Model Performance"])

if menu == "Data Overview":
    st.header("1. Understanding the Dataset")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df_raw))
    c2.metric("Pollutant Types", df_raw['pollutant_id'].nunique())
    c3.metric("Avg Pollutant Level", round(df_raw['pollutant_avg'].mean(), 2))

    st.markdown("""
    <div class="result-box">
        <h4>Initial Code Outcome:</h4>
        <p>Our code started by cleaning the data. We removed any rows where the pollution level was missing. 
        As seen in the data sample below, we are tracking different chemicals (PM2.5, NO2, SO2) across various GPS coordinates in India.</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

elif menu == "Location & Interaction Results":
    st.header("2. How Location Impacts Pollution")
    
    st.markdown("""
    <div class="result-box">
        <h4>Key Code Results:</h4>
        <p>In our notebook, we didn't just use basic Latitude and Longitude. We calculated <b>Squared Values</b> and <b>Interactions</b>.</p>
        <ul>
            <li><b>Latitude² / Longitude²:</b> This helped the code identify specific "zones" or hotspots of pollution rather than just assuming pollution increases in a straight line.</li>
            <li><b>Lat * Long Interaction:</b> This captured how the combination of North-South and East-West positions creates unique regional air quality patterns.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Visualization of Lat/Long relation
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    sns.scatterplot(data=df_raw, x='longitude', y='latitude', hue='pollutant_avg', palette='hot', ax=ax)
    st.pyplot(fig)

elif menu == "Final Model Performance":
    st.header("3. Accuracy & Statistical Outcomes")
    
    # Run the OLS Model from Cell 16/23
    formula = 'pollutant_avg ~ latitude + longitude + latitude2 + longitude2 + lat_long_interaction + pollutant_id_NH3 + pollutant_id_NO2 + pollutant_id_OZONE + pollutant_id_PM10 + pollutant_id_PM2_5 + pollutant_id_SO2'
    model = smf.ols(formula, data=df_p).fit()
    
    st.markdown(f"""
    <div class="result-box">
        <h4>The Final Prediction Outcome:</h4>
        <p>The model reached an <b>R-Squared of {model.rsquared:.2f}</b>.</p>
        <p>This means your code can explain <b>{model.rsquared*100:.0f}%</b> of the changes in air quality using only the location and the type of pollutant.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Statistical Result Table")
    st.text(str(model.summary().tables[1]))

    # K-Fold Accuracy (Cell 24)
    X = df_p[['latitude', 'longitude', 'latitude2', 'longitude2', 'lat_long_interaction', 
              'pollutant_id_NH3', 'pollutant_id_NO2', 'pollutant_id_OZONE', 
              'pollutant_id_PM10', 'pollutant_id_PM2_5', 'pollutant_id_SO2']]
    y = df_p['pollutant_avg']
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.sqrt(np.mean(-scores))

    st.success(f"Final Validation: The model's average error is {rmse:.2f} units. This proves the analysis is consistent and reliable.")