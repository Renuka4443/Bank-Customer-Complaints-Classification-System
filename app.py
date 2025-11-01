import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

# Page config
st.set_page_config(
    page_title="Bank Complaint Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url('https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    }
    
    
    .stButton>button {
        background: linear-gradient(90deg, #2196F3 0%, #64B5F6 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.5);
        background: linear-gradient(90deg, #1976D2 0%, #42A5F5 100%);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(33, 150, 243, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(33, 150, 243, 0.2) !important;
        min-width: 250px !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        margin-bottom: 8px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 8px 12px !important;
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(33, 150, 243, 0.2) !important;
        border-radius: 10px !important;
        color: #2196F3 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    [data-testid="stSidebar"] .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(33, 150, 243, 0.1), transparent);
        transition: left 0.5s;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover::before {
        left: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(33, 150, 243, 0.15) !important;
        border-color: rgba(33, 150, 243, 0.4) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2) !important;
    }
    
    
    div[data-baseweb="select"] > div {
        min-height: 32px !important;
        height: 32px !important;
    }
    
    div[data-baseweb="select"] > div > div {
        padding: 6px 12px !important;
        font-size: 13px !important;
    }
    
    div[data-baseweb="select"] [data-baseweb="select"] {
        font-size: 13px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text for prediction"""
    if pd.isna(text) or not str(text).strip():
        return ''
    text = str(text).lower().strip()
    
    # Remove URLs, emails, special characters, digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    
    # Lemmatize
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]
    
    # Remove short words
    words = [w for w in words if len(w) > 2]
    
    return ' '.join(words)

# Load models (cached)
@st.cache_data
def load_model_d1(model_name):
    """Load Dataset 1 models"""
    if model_name == 'Logistic Regression':
        model = joblib.load('logistic_model_D1.pkl')
    else:
        model = joblib.load('svm_model_D1.pkl')
    return model

@st.cache_data
def load_model_d2(model_name):
    """Load Dataset 2 models"""
    if model_name == 'Logistic Regression':
        model = joblib.load('logistic_model_D2.pkl')
    else:
        model = joblib.load('svm_model_D2.pkl')
    return model

@st.cache_data
def load_vectorizer_d1():
    """Load Dataset 1 vectorizer"""
    return joblib.load('tfidf_vectorizer_D1.pkl')

@st.cache_data
def load_vectorizer_d2():
    """Load Dataset 2 vectorizer"""
    return joblib.load('tfidf_vectorizer_D2.pkl')

@st.cache_data
def load_encoder_d1():
    """Load Dataset 1 label encoder"""
    return joblib.load('label_encoder_D1.pkl')

@st.cache_data
def load_encoder_d2():
    """Load Dataset 2 label encoder"""
    return joblib.load('label_encoder_D2.pkl')

# Helper function to get icon for category
def get_category_icon(category_name):
    """Returns the icon class for a given category name."""
    # Normalize the category name by removing underscores and case-insensitive matching
    normalized_name = category_name.lower().replace('_', ' ').strip()
    
    icon_mapping = {
        'credit card': 'ri-bank-card-line',
        'retail banking': 'ri-store-3-line',
        'credit report': 'ri-bar-chart-line',
        'credit reporting': 'ri-bar-chart-line',
        'mortgages': 'ri-home-line',
        'mortgages & loans': 'ri-home-line',
        'mortgage': 'ri-home-line',
        'debt collection': 'ri-phone-line',
        'loan': 'ri-money-dollar-circle-line',
        'bank account': 'ri-bank-line'
    }
    
    # Try exact match
    if normalized_name in icon_mapping:
        return icon_mapping[normalized_name]
    
    # Try partial matching for categories that might have extra words
    for key in icon_mapping:
        if key in normalized_name or normalized_name in key:
            return icon_mapping[key]
    
    # Default icon
    return 'ri-checkbox-circle-fill'

# Session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "dataset1"

def main():
    # Sidebar navigation
    with st.sidebar:
        # Header
        st.markdown("""
            <div style="text-align: center; margin-bottom: 25px; padding: 15px; 
                        background: rgba(255, 255, 255, 0.3); backdrop-filter: blur(15px);
                        border-radius: 15px; border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.1);">
                <i class="ri-compass-3-line" style="font-size: 26px; color: #2196F3; 
                         display: inline-block; margin-right: 8px; vertical-align: middle;"></i>
                <span style="color: #2196F3; font-weight: 700; font-size: 16px;">Navigation Menu</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons with icons beside them
        nav_options = [
            ("ri-database-2-line", "Dataset 1", "dataset1"),
            ("ri-server-line", "Dataset 2", "dataset2"),
            ("ri-file-copy-2-line", "Samples", "samples"),
            ("ri-bar-chart-box-line", "Insights", "insights"),
            ("ri-information-line", "About", "about")
        ]
        
        for icon, label, page_key in nav_options:
            selected = "✓ " if st.session_state.current_page == page_key else ""
            
            # Display icon beside button
            col_icon, col_btn = st.columns([0.3, 2.7])
            with col_icon:
                st.markdown(f"""
                    <div style="text-align: right; padding-top: 10px;">
                        <i class="{icon}" style="font-size: 22px; color: #2196F3;"></i>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_btn:
                if st.button(f"{selected}{label}", use_container_width=True, key=f"nav_{page_key}"):
                    st.session_state.current_page = page_key
                    st.rerun()
    
    # Main content area - compact header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 8px;">
            <h1 style="color: #2196F3; font-weight: 800; font-size: 28px; margin: 0;">
                <i class="ri-bank-line" style="font-size: 30px; vertical-align: middle; margin-right: 8px;"></i>
                Your Smart Bank Complaint Classifier
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.session_state.current_page
    
    if page == "dataset1" or page == "📊 Dataset 1 Prediction":
        dataset1_page()
    elif page == "dataset2" or page == "📈 Dataset 2 Prediction":
        dataset2_page()
    elif page == "samples" or page == "📋 Samples":
        samples_page()
    elif page == "insights" or page == "📊 Insights":
        insights_page()
    elif page == "about" or page == "ℹ️ About":
        about_page()

def dataset1_page():
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);
                    transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);
                            padding: 8px; border-radius: 10px; box-shadow: 0 3px 10px rgba(33, 150, 243, 0.3);">
                    <i class="ri-message-3-fill" style="font-size: 18px; color: white;"></i>
                </div>
                <div>
                    <h2 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                        Dataset 1 Classification
                    </h2>
                </div>
            </div>
            <p style="color: #555; font-size: 12px; line-height: 1.5; margin: 0; padding: 8px;
                      background: rgba(33, 150, 243, 0.05); border-radius: 8px;">
                <i class="ri-information-line" style="margin-right: 5px; color: #2196F3;"></i>
                Enter a bank complaint to automatically classify it into one of five categories.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Category information with enhanced styling
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                <i class="ri-list-check-2" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                    Available Categories
                </h3>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-bank-card-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Credit Card</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-store-3-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Retail Banking</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-bar-chart-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Credit Report</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-home-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Mortgages</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-phone-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Debt Collection</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection with enhanced styling
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <i class="ri-cpu-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 700; margin: 0; font-size: 14px;">
                    Select ML Model
                </h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Support Vector Machine"],
        key="model_d1",
        label_visibility="collapsed"
    )
    
    # Normalize model name
    if "Logistic" in model_choice:
        model_choice = "Logistic Regression"
    else:
        model_choice = "Support Vector Machine"
    
    # Text input with enhanced styling
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <i class="ri-file-edit-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 700; margin: 0; font-size: 14px;">
                    Enter Your Complaint
                </h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    complaint_text = st.text_area(
        "Enter Your Complaint",
        height=110,
        placeholder="Example: I was charged an annual fee on my credit card even though the bank representative assured me it was a lifetime free card. I have been trying to reach customer service for a week but nobody has responded to my emails or calls...",
        key="complaint_d1",
        label_visibility="collapsed"
    )
    
    # Predict button - using Streamlit button with icon emoji
    st.markdown("""
        <div style="text-align: center; margin: 15px 0;">
            <style>
            #root .stButton > button {
                display: inline-flex !important;
                align-items: center !important;
                gap: 10px !important;
            }
            </style>
        </div>
    """, unsafe_allow_html=True)
    
    button_clicked = st.button("🔍 Analyze Complaint", key="predict_d1", use_container_width=True)
    
    if button_clicked:
        if not complaint_text.strip():
            st.error("⚠️ Please enter a complaint before predicting!")
        else:
            with st.spinner("Analyzing complaint..."):
                # Clean text
                cleaned_text = clean_text(complaint_text)
                
                if not cleaned_text:
                    st.warning("⚠️ The complaint text could not be processed. Please enter a more descriptive complaint.")
                else:
                    # Load models
                    model = load_model_d1(model_choice)
                    vectorizer = load_vectorizer_d1()
                    encoder = load_encoder_d1()
                    
                    # Transform and predict
                    text_vector = vectorizer.transform([cleaned_text])
                    prediction = model.predict(text_vector)
                    category = encoder.inverse_transform(prediction)[0]
                    
                    # Format category name
                    formatted_category = category.replace('_', ' ').title()
                    category_icon = get_category_icon(formatted_category)
                    
                    # Display result with enhanced styling
                    st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px);
                                    border: 2px solid rgba(33, 150, 243, 0.3); border-radius: 18px;
                                    padding: 25px; margin-top: 15px; text-align: center;
                                    box-shadow: 0 12px 40px rgba(33, 150, 243, 0.2);
                                    position: relative; overflow: hidden; animation: resultPulse 0.6s ease-out;">
                            <div style="font-size: 48px; margin-bottom: 12px; animation: bounceScale 0.8s ease-out;">
                                <i class="{category_icon}" style="color: #2196F3; filter: drop-shadow(0 8px 16px rgba(33, 150, 243, 0.4));"></i>
                            </div>
                            <div style="font-size: 12px; margin-bottom: 8px; color: #2196F3; font-weight: 500; letter-spacing: 1.5px;">
                                PREDICTED CATEGORY
                            </div>
                            <div style="font-size: 26px; font-weight: 800; color: #2196F3;">
                                {formatted_category}
                            </div>
                            <div style="margin-top: 12px; font-size: 12px; color: #666; padding: 8px;
                                        background: rgba(33, 150, 243, 0.05); border-radius: 8px; backdrop-filter: blur(10px);">
                                <i class="ri-cpu-line" style="margin-right: 5px; color: #2196F3;"></i>
                                Model: {model_choice}
                            </div>
                        </div>
                        <style>
                        @keyframes resultPulse {{
                            0% {{ opacity: 0; transform: scale(0.9) translateY(20px); }}
                            100% {{ opacity: 1; transform: scale(1) translateY(0); }}
                        }}
                        @keyframes bounceScale {{
                            0%, 100% {{ transform: scale(1); }}
                            50% {{ transform: scale(1.15); }}
                        }}
                        </style>
                    """, unsafe_allow_html=True)

def dataset2_page():
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);
                    transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);
                            padding: 8px; border-radius: 10px; box-shadow: 0 3px 10px rgba(33, 150, 243, 0.3);">
                    <i class="ri-message-3-fill" style="font-size: 18px; color: white;"></i>
                </div>
                <div>
                    <h2 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                        Dataset 2 Classification
                    </h2>
                </div>
            </div>
            <p style="color: #555; font-size: 12px; line-height: 1.5; margin: 0; padding: 8px;
                      background: rgba(33, 150, 243, 0.05); border-radius: 8px;">
                <i class="ri-information-line" style="margin-right: 5px; color: #2196F3;"></i>
                Enter a bank complaint to automatically classify it into one of six categories.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Category information with enhanced styling
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                <i class="ri-list-check-2" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                    Available Categories
                </h3>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-bar-chart-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Credit Report</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-money-dollar-circle-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Loan</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-phone-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Debt Collection</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-home-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Mortgage</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-bank-card-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Credit Card</span>
                </div>
                <div class="category-card" style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                                                    backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                                                    padding: 8px 12px; border-radius: 8px; transition: all 0.3s ease;
                                                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <div style="font-size: 20px; margin-bottom: 4px;">
                        <i class="ri-bank-line" style="color: #2196F3;"></i>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: #2196F3;">Bank Account</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection with enhanced styling
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <i class="ri-cpu-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 700; margin: 0; font-size: 14px;">
                    Select ML Model
                </h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Support Vector Machine"],
        key="model_d2",
        label_visibility="collapsed"
    )
    
    # Text input with enhanced styling
    st.markdown("""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <i class="ri-file-edit-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 700; margin: 0; font-size: 14px;">
                    Enter Your Complaint
                </h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    complaint_text = st.text_area(
        "Enter Your Complaint",
        height=110,
        placeholder="Example: My mortgage payment was deducted twice from my account this month, and despite multiple calls, the refund hasn't been processed yet...",
        key="complaint_d2",
        label_visibility="collapsed"
    )
    
    # Predict button - using Streamlit button with icon emoji
    button_clicked_d2 = st.button("🔍 Analyze Complaint", key="predict_d2", use_container_width=True)
    
    if button_clicked_d2:
        if not complaint_text.strip():
            st.error("⚠️ Please enter a complaint before predicting!")
        else:
            with st.spinner("Analyzing complaint..."):
                # Clean text
                cleaned_text = clean_text(complaint_text)
                
                if not cleaned_text:
                    st.warning("⚠️ The complaint text could not be processed. Please enter a more descriptive complaint.")
                else:
                    # Load models
                    model = load_model_d2(model_choice)
                    vectorizer = load_vectorizer_d2()
                    encoder = load_encoder_d2()
                    
                    # Transform and predict
                    text_vector = vectorizer.transform([cleaned_text])
                    prediction = model.predict(text_vector)
                    category = encoder.inverse_transform(prediction)[0]
                    
                    # Get icon for category
                    category_icon = get_category_icon(category)
                    
                    # Display result with enhanced styling
                    st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px);
                                    border: 2px solid rgba(33, 150, 243, 0.3); border-radius: 18px;
                                    padding: 25px; margin-top: 15px; text-align: center;
                                    box-shadow: 0 12px 40px rgba(33, 150, 243, 0.2);
                                    position: relative; overflow: hidden; animation: resultPulse 0.6s ease-out;">
                            <div style="font-size: 48px; margin-bottom: 12px; animation: bounceScale 0.8s ease-out;">
                                <i class="{category_icon}" style="color: #2196F3; filter: drop-shadow(0 8px 16px rgba(33, 150, 243, 0.4));"></i>
                            </div>
                            <div style="font-size: 12px; margin-bottom: 8px; color: #2196F3; font-weight: 500; letter-spacing: 1.5px;">
                                PREDICTED CATEGORY
                            </div>
                            <div style="font-size: 26px; font-weight: 800; color: #2196F3;">
                                {category}
                            </div>
                            <div style="margin-top: 12px; font-size: 12px; color: #666; padding: 8px;
                                        background: rgba(33, 150, 243, 0.05); border-radius: 8px; backdrop-filter: blur(10px);">
                                <i class="ri-cpu-line" style="margin-right: 5px; color: #2196F3;"></i>
                                Model: {model_choice}
                            </div>
                        </div>
                        <style>
                        @keyframes resultPulse {{
                            0% {{ opacity: 0; transform: scale(0.9) translateY(20px); }}
                            100% {{ opacity: 1; transform: scale(1) translateY(0); }}
                        }}
                        @keyframes bounceScale {{
                            0%, 100% {{ transform: scale(1); }}
                            50% {{ transform: scale(1.15); }}
                        }}
                        </style>
                    """, unsafe_allow_html=True)

def about_page():
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 10px; padding: 10px; border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1); text-align: center;">
                <div style="font-size: 26px; margin-bottom: 6px; color: #2196F3;">
                    <i class="ri-database-line"></i>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #2196F3; margin-bottom: 6px; text-transform: uppercase;">
                    Datasets
                </div>
                <div style="font-size: 28px; font-weight: 800; background: linear-gradient(135deg, #2196F3, #64B5F6); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    2
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 10px; padding: 10px; border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1); text-align: center;">
                <div style="font-size: 26px; margin-bottom: 6px; color: #2196F3;">
                    <i class="ri-brain-line"></i>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #2196F3; margin-bottom: 6px; text-transform: uppercase;">
                    ML Models
                </div>
                <div style="font-size: 28px; font-weight: 800; background: linear-gradient(135deg, #2196F3, #64B5F6); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    4
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 10px; padding: 10px; border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1); text-align: center;">
                <div style="font-size: 26px; margin-bottom: 6px; color: #2196F3;">
                    <i class="ri-file-text-line"></i>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #2196F3; margin-bottom: 6px; text-transform: uppercase;">
                    Complaints
                </div>
                <div style="font-size: 28px; font-weight: 800; background: linear-gradient(135deg, #2196F3, #64B5F6); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    186K+
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 10px; padding: 10px; border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1); text-align: center;">
                <div style="font-size: 26px; margin-bottom: 6px; color: #2196F3;">
                    <i class="ri-focus-3-line"></i>
                </div>
                <div style="font-size: 12px; font-weight: 600; color: #2196F3; margin-bottom: 6px; text-transform: uppercase;">
                    Categories
                </div>
                <div style="font-size: 28px; font-weight: 800; background: linear-gradient(135deg, #2196F3, #64B5F6); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    11
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # About Us card
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);
                    transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);
                            padding: 8px; border-radius: 10px; box-shadow: 0 3px 10px rgba(33, 150, 243, 0.3);">
                    <i class="ri-information-fill" style="font-size: 18px; color: white;"></i>
                </div>
                <div>
                    <h2 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 18px;">
                        About the Platform
                    </h2>
                </div>
            </div>
            <p style="color: #555; font-size: 14px; line-height: 1.6; margin: 0; padding: 10px;
                      background: rgba(33, 150, 243, 0.05); border-radius: 8px;">
                <i class="ri-information-line" style="margin-right: 6px; color: #2196F3;"></i>
               We are a team dedicated to improving customer service in the banking sector by creating solutions that make complaint handling faster and more efficient. Our project helps banks quickly understand and address customer concerns by classifying complaints into proper categories, ensuring they are directed to the right departments for timely resolution and greater customer satisfaction.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.3);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.15);">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                <i class="ri-database-2-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                    Dataset Information
                </h3>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                            backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                            padding: 10px 14px; border-radius: 8px; transition: all 0.3s ease;
                            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 13px; font-weight: 700; color: #2196F3; margin: 0;">Dataset 1: 162,421 complaints</p>
                </div>
                <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(118, 75, 162, 0.1)); 
                            backdrop-filter: blur(10px); border: 2px solid rgba(33, 150, 243, 0.3);
                            padding: 10px 14px; border-radius: 8px; transition: all 0.3s ease;
                            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 13px; font-weight: 700; color: #2196F3; margin: 0;">Dataset 2: 24,374 complaints</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ML Models
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.3);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.15);">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 10px;">
                <i class="ri-cpu-line" style="font-size: 18px; color: #2196F3;"></i>
                <h3 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                    ML Models & Technology
                </h3>
            </div>
            <p style="color: #555; line-height: 1.7; margin: 0; font-size: 14px;">
                Uses <strong>Logistic Regression</strong> and <strong>Support Vector Machine</strong> with TF-IDF 
                vectorization and SMOTE for handling class imbalance. Both models achieve high accuracy in classifying 
                customer complaints into relevant categories.
            </p>
        </div>
    """, unsafe_allow_html=True)

def insights_page():
    import plotly.graph_objects as go
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);
                    transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);
                            padding: 8px; border-radius: 10px; box-shadow: 0 3px 10px rgba(33, 150, 243, 0.3);">
                    <i class="ri-bar-chart-box-fill" style="font-size: 18px; color: white;"></i>
                </div>
                <div>
                    <h2 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                        Model Performance Insights
                    </h2>
                </div>
            </div>
            <p style="color: #555; font-size: 12px; line-height: 1.5; margin: 0; padding: 8px;
                      background: rgba(33, 150, 243, 0.05); border-radius: 8px;">
                <i class="ri-information-line" style="margin-right: 5px; color: #2196F3;"></i>
                Evaluation metrics and comparison of machine learning models.
            </p>
        </div>
        
        <style>
        .insights .metric-bar {
            background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%);
            height: 100%;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 13px;
            transition: all 0.3s ease;
        }
        .insights .metric-bar:hover {
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        }
        .insights .bar-container {
            background: rgba(33, 150, 243, 0.1);
            height: 24px;
            border-radius: 4px;
            margin: 6px 0;
            position: relative;
            overflow: hidden;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Wrap content in insights class
    st.markdown('<div class="insights">', unsafe_allow_html=True)
    
    # Dataset 1 Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 12px; padding: 12px; margin-bottom: 10px;
                        border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);">
                <h3 style="color: #2196F3; font-weight: 800; margin: 0 0 12px 0; font-size: 16px;">
                    <i class="ri-database-2-line" style="margin-right: 6px;"></i>Dataset 1
                </h3>
                <div style="background: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 12px;
                            border-left: 4px solid #4CAF50;">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Best Model: Logistic Regression</div>
                    <div style="font-size: 18px; font-weight: 800; color: #4CAF50;">86.15% Accuracy</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create bar chart for Dataset 1
        st.markdown("#### Model Comparison - Dataset 1")
        fig1 = go.Figure(data=[
            go.Bar(name='Logistic Regression', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                   y=[86.15, 87.17, 86.15, 86.44], marker_color='#66BB6A', width=[0.3, 0.3, 0.3, 0.3],
                   text=[86.15, 87.17, 86.15, 86.44], texttemplate='%{text:.2f}%', textposition='outside', textfont=dict(size=11)),
            go.Bar(name='Support Vector Machine', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                   y=[85.45, 86.48, 85.45, 85.76], marker_color='#4CAF50', width=[0.3, 0.3, 0.3, 0.3],
                   text=[85.45, 86.48, 85.45, 85.76], texttemplate='%{text:.2f}%', textposition='outside', textfont=dict(size=11))
        ])
        fig1.update_layout(barmode='group', height=280, margin=dict(l=10, r=10, t=40, b=30),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(size=11), showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
                           dragmode=False, hovermode='x unified')
        fig1.update_xaxes(gridcolor='rgba(33, 150, 243, 0.2)', showgrid=True)
        fig1.update_yaxes(gridcolor='rgba(33, 150, 243, 0.2)', showgrid=True, range=[82, 90])
        fig1.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                        border-radius: 12px; padding: 12px; margin-bottom: 10px;
                        border: 2px solid rgba(33, 150, 243, 0.2);
                        box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);">
                <h3 style="color: #2196F3; font-weight: 800; margin: 0 0 12px 0; font-size: 16px;">
                    <i class="ri-server-line" style="margin-right: 6px;"></i>Dataset 2
                </h3>
                <div style="background: rgba(33, 150, 243, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 12px;
                            border-left: 4px solid #2196F3;">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Best Model: Logistic Regression</div>
                    <div style="font-size: 18px; font-weight: 800; color: #2196F3;">85.54% Accuracy</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create bar chart for Dataset 2
        st.markdown("#### Model Comparison - Dataset 2")
        fig2 = go.Figure(data=[
            go.Bar(name='Logistic Regression', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                   y=[85.54, 85.63, 85.54, 85.57], marker_color='#2196F3', width=[0.3, 0.3, 0.3, 0.3],
                   text=[85.54, 85.63, 85.54, 85.57], texttemplate='%{text:.2f}%', textposition='outside', textfont=dict(size=11)),
            go.Bar(name='Support Vector Machine', x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                   y=[84.80, 84.86, 84.80, 84.82], marker_color='#64B5F6', width=[0.3, 0.3, 0.3, 0.3],
                   text=[84.80, 84.86, 84.80, 84.82], texttemplate='%{text:.2f}%', textposition='outside', textfont=dict(size=11))
        ])
        fig2.update_layout(barmode='group', height=280, margin=dict(l=10, r=10, t=40, b=30),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(size=11), showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
                           dragmode=False, hovermode='x unified')
        fig2.update_xaxes(gridcolor='rgba(33, 150, 243, 0.2)', showgrid=True)
        fig2.update_yaxes(gridcolor='rgba(33, 150, 243, 0.2)', showgrid=True, range=[81, 87])
        fig2.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    # Close insights div
    st.markdown('</div>', unsafe_allow_html=True)

def samples_page():
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); 
                    border-radius: 12px; padding: 12px; margin-bottom: 10px;
                    border: 2px solid rgba(33, 150, 243, 0.2);
                    box-shadow: 0 6px 25px rgba(33, 150, 243, 0.1);
                    transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);
                            padding: 8px; border-radius: 10px; box-shadow: 0 3px 10px rgba(33, 150, 243, 0.3);">
                    <i class="ri-file-copy-2-line" style="font-size: 18px; color: white;"></i>
                </div>
                <div>
                    <h2 style="color: #2196F3; font-weight: 800; margin: 0; font-size: 16px;">
                        Sample Inputs & Expected Categories
                    </h2>
                </div>
            </div>
            <p style="color: #555; font-size: 12px; line-height: 1.5; margin: 0; padding: 8px;
                      background: rgba(33, 150, 243, 0.05); border-radius: 8px;">
                <i class="ri-information-line" style="margin-right: 5px; color: #2196F3;"></i>
                Copy any sample complaint below and paste it into the Dataset 1 or Dataset 2 page to try the app.
            </p>
        </div>
    """, unsafe_allow_html=True)

    d1_samples = [
        ("Credit card", "I was charged an annual fee on my credit card even though it was supposed to be lifetime free. Support hasn’t resolved it and late fees were added."),
        ("Bank account", "My checking account shows an unauthorized debit and the bank hasn’t reversed it despite multiple complaints."),
        ("Debt collection", "Debt collector keeps calling me about a loan I never took. They’re threatening legal action."),
        ("Credit reporting", "My credit report shows an account I don’t recognize and my dispute hasn’t been processed."),
        ("Mortgages", "Mortgage escrow was miscalculated and my monthly payment increased without clear explanation.")
    ]

    d2_samples = [
        ("Mortgages", "My mortgage payment was deducted twice this month and I still haven’t received a refund. This caused an overdraft fee."),
        ("Credit card", "My credit card statement has a fraudulent online purchase. The issuer declined my dispute."),
        ("Debt collection", "Debt collectors are calling my workplace and family about an old medical bill."),
        ("Bank account", "A savings account transfer is stuck and the bank can’t locate the funds."),
        ("Credit reporting", "The credit bureau hasn’t removed an inaccurate late payment despite submitting proof.")
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Dataset 1")
        for label, text in d1_samples:
            st.markdown(f"**Expected category**: {label}")
            st.code(text)
    with c2:
        st.markdown("### Dataset 2")
        for label, text in d2_samples:
            st.markdown(f"**Expected category**: {label}")
            st.code(text)

if __name__ == "__main__":
    main()

