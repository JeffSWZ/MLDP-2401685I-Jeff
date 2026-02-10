# Import required libraries
import streamlit as st
import pandas as pd
import joblib
import base64
from datetime import datetime

# Configure Streamlit page settings
st.set_page_config(
    page_title="Bike Purchase Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Convert background image to base64 so it can be embedded in CSS
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load and encode background image
bg_image = get_base64_image("background.jpg")

# Apply custom CSS styling for background, panels, inputs, and buttons
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-color: rgba(3, 7, 18, 0.85);
        z-index: 0;
        pointer-events: none;
    }}

    section[data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
    }}

    section[data-testid="stSidebar"] {{
        background-color: rgba(3, 7, 18, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    h1, h2, h3, p {{
        color: #f8fafc;
    }}


    .panel {{
        background: rgba(30, 41, 59, 0.92);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }}

    .stSelectbox > div > div,
    .stNumberInput input {{
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }}

    .stNumberInput input {{
        height: 44px !important;
        padding: 0.45rem 0.75rem !important;
        font-size: 0.95rem !important;
    }}

    .stButton > button {{
        background-color: #2563eb;
        color: #f8fafc;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: #3b82f6;
    }}

    .result-yes {{
        background: linear-gradient(135deg, #064e3b, #022c22);
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
    }}

    .result-no {{
        background: linear-gradient(135deg, #7f1d1d, #450a0a);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 18px;
        padding: 2.2rem;
        text-align: center;
    }}

    .result-title {{
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }}

    .result-prob {{
        font-size: 1.2rem;
        color: #e5e7eb;
    }}

    .placeholder {{
        color: #94a3b8;
        font-style: italic;
        text-align: center;
        padding: 2rem 1rem;
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 14px;
        margin-top: 1rem;
    }}

    .footer {{
        text-align: center;
        color: #c7d2fe;
        margin-top: 2rem;
    }}

    a[href^="#"] {{
    display: none;
    }}


    
    </style>
    """,
    unsafe_allow_html=True
)

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Load trained Random Forest model
model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select page", ["Dashboard", "Predict Purchase"])

# Prepare user input to match the model's feature format
def prepare_model_input(
    income, children, cars, marital_status, gender,
    education, occupation, home_owner,
    commute_distance, region, age_group
):
    # Calculate income per person
    income_per_person = income / (children + 1)

    # Create a dictionary for the input features
    row = {
        "Income": income,
        "Children": children,
        "Cars": cars,
        "income_per_person": income_per_person,
        "Marital Status_Single": marital_status == "Single",
        "Gender_Male": gender == "Male",
        "Education_Graduate Degree": education == "Graduate Degree",
        "Education_High School": education == "High School",
        "Education_Partial College": education == "Partial College",
        "Education_Partial High School": education == "Partial High School",
        "Occupation_Manual": occupation == "Manual",
        "Occupation_Professional": occupation == "Professional",
        "Occupation_Skilled Manual": occupation == "Skilled Manual",
        "Occupation_Management": occupation == "Management",
        "Home Owner_Yes": home_owner == "Yes",
        "Commute Distance_1-2 Miles": commute_distance == "1-2 Miles",
        "Commute Distance_2-5 Miles": commute_distance == "2-5 Miles",
        "Commute Distance_5-10 Miles": commute_distance == "5-10 Miles",
        "Commute Distance_10+ Miles": commute_distance == "10+ Miles",
        "Region_North America": region == "North America",
        "Region_Pacific": region == "Pacific",
        "age_group_30-40": age_group == "30-40",
        "age_group_40-50": age_group == "40-50",
        "age_group_50-60": age_group == "50-60",
        "age_group_60+": age_group == "60+"
    }

    # Convert to DataFrame and reindex to match model features
    return pd.DataFrame([row]).reindex(
        columns=model.feature_names_in_,
        fill_value=False
    )

# Dashboard page
if page == "Dashboard":

    # Display dashboard metrics and overview
    st.title("Bike Purchase Prediction Dashboard")

    # Display model metrics in three columns
    col1, col2 = st.columns(2)
    col1.metric("Model Type", type(model).__name__)
    col2.metric("Total Features", len(model.feature_names_in_))

    st.markdown(
    """
    <div class="panel">
        <h3>System Overview</h3>
        <p>
        To predict whether a customer will purchase a bike based on their demographic
        and lifestyle information, allowing businesses to target high-probability
        customers more effectively. Through this predictive approach, companies can
        reduce advertising expenses and lower customer acquisition costs.
        </p>
        <h3>Task</h3>
        <p>Classification - Predict whether a customer will purchase a bike (Yes/No)</p>
    </div>
    """,
    unsafe_allow_html=True
)



# Prediction page
elif page == "Predict Purchase":
    st.title("Predict Bike Purchase")

    # Input form and prediction result side by side
    col1, col2 = st.columns([1.1, 1])

    # Input form
    with col1:
        income = st.number_input("Income", min_value=0, step=1000) # Ensure non-negative income
        age_group = st.selectbox("Age group", ["30-40", "40-50", "50-60", "60+"]) 
        children = st.slider("Number of children", 0, 5)
        cars = st.slider("Number of cars", 0, 4)
        marital_status = st.selectbox("Marital status", ["Single", "Married"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education level", ["Graduate Degree", "High School", "Partial College", "Partial High School"])
        occupation = st.selectbox("Occupation", ["Manual", "Professional", "Skilled Manual", "Management"])
        home_owner = st.selectbox("Home owner", ["Yes", "No"])
        commute_distance = st.selectbox("Commute distance", ["0-1 Miles", "1-2 Miles", "2-5 Miles", "5-10 Miles", "10+ Miles"])
        region = st.selectbox("Region", ["North America", "Pacific"])
        predict_btn = st.button("Run Prediction")

    # Prediction result
    with col2:
        st.subheader("Prediction Result")

        # Placeholder before prediction
        if not predict_btn:
            st.markdown(
                """
                <div class="placeholder">
                    Prediction result will appear here
                </div>
                """,
                unsafe_allow_html=True
            )

        # Run prediction when button is clicked
        if predict_btn:
            commute_key = "1-2 Miles" if commute_distance == "0-1 Miles" else commute_distance

            # Prepare model input
            X_new = prepare_model_input(
                income, children, cars, marital_status, gender,
                education, occupation, home_owner,
                commute_key, region, age_group
            )

            # Get prediction and probability
            prob = model.predict_proba(X_new)[0, 1]
            pred = prob >= 0.5

            # Display prediction result, Yes or No with probability
            if pred:
                st.markdown(
                    f"""
                    <div class="result-yes">
                        <div class="result-title">YES</div>
                        <div class="result-prob">Probability of Purchase {prob:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-no">
                        <div class="result-title">NO</div>
                        <div class="result-prob">Probability of Purchase {prob:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# Footer
st.markdown(
    """
    <p class="footer">
    Bike Purchase Prediction System
    </p>
    """,
    unsafe_allow_html=True
)
