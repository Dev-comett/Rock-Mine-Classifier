import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Page Config & Styling ---
st.set_page_config(page_title="Rock🪨 vs Mine💣 Classifier", layout="centered")
st.markdown(
    """
    <style>
    .main {background-color: #F5F5F5; padding: 2rem; border-radius: 1rem; margin-bottom: 4rem;}
    .title {font-size: 3rem; color: #333333; font-weight: bold; margin-bottom: 1rem;}
    .footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: #333333; color: white; text-align: center; padding: 1rem;}
    a {color: #1E88E5;}
    </style>
    """, unsafe_allow_html=True
)

# --- Utility Functions ---
@st.cache_data
def load_sonar_dataset(path=None, url=None, csv_buffer=None):
    if csv_buffer is not None:
        df = pd.read_csv(csv_buffer)
    elif path is not None:
        df = pd.read_csv(path, header=None)
    else:
        df = pd.read_csv(url, header=None)
    df.columns = [f"feature_{i+1}" for i in range(df.shape[1]-1)] + ['label']
    return df

@st.cache_resource
def train_sonar_model(df):
    X = df.iloc[:, :-1].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, train_acc, test_acc

# --- Main App Layout ---

st.markdown('<div class="title">🚀 Rock🪨 vs Mine💣 Classifier</div>', unsafe_allow_html=True)
st.write("This app predicts whether sonar signals reflect a Rock or a Mine.")

# --- Dataset Section ---
st.subheader("📊 Dataset")
# Check for local CSV
LOCAL_FILE = "sonar data.csv"
if os.path.exists(LOCAL_FILE):
    data = load_sonar_dataset(path=LOCAL_FILE)
    st.info(f"Loaded local dataset from **{LOCAL_FILE}**")
else:
    DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/sonar.csv"
    st.markdown(f"Download the original Sonar dataset: [CSV]({DATA_URL})")
    data = load_sonar_dataset(url=DATA_URL)

if st.checkbox("Show raw dataset sample"):
    st.dataframe(data.head())

# --- Custom Upload ---
st.subheader("📂 Upload Your Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV with 60 features + 'label' column", type=["csv"])
if uploaded_file:
    custom_df = load_sonar_dataset(csv_buffer=uploaded_file)
    if 'label' not in custom_df.columns or custom_df.shape[1] < 61:
        st.warning("Ensure your dataset has 60 feature columns and a 'label' column named exactly 'label'.")
    else:
        data = custom_df
        st.success("Custom dataset loaded!")
        st.dataframe(data.head())

# --- Model Training & Metrics ---
st.subheader("⚙️ Model Training & Metrics")
model, train_acc, test_acc = train_sonar_model(data)
st.write(f"**Training Accuracy:** {train_acc:.2f}")
st.write(f"**Test Accuracy:** {test_acc:.2f}")

# --- Prediction Section ---
st.selectbox("🔮 Make a Prediction")
st.write("Enter values for 60 features below:")
user_input = []
cols = st.columns(3)
for i in range(60):
    user_input.append(
        cols[i % 3].number_input(f"F{i+1}", value=0.0, format="%.5f")
    )

if st.button("Predict"):
    arr = np.array(user_input).reshape(1, -1)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr).max()
    label = "Rock 🪨" if pred == 'R' else "Mine 💣"
    if pred == 'R':
        st.success(f"Prediction: **{label}** (Confidence: {prob:.2f})")
    else:
        st.error(f"Prediction: **{label}** (Confidence: {prob:.2f})")

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
footer_html = '''
<div class="footer">
    © 2025 Devansh Mishra |
    <a href="https://www.linkedin.com/in/dev-ice/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/Dev-comett" target="_blank">GitHub</a>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)

