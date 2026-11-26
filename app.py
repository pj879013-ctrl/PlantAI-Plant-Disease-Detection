
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from pathlib import Path
from database import (
    create_table,
    save_prediction,
    get_all_predictions,
    clear_predictions
)
from PIL import Image
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantAI | Smart Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
IMAGE_SIZE = (160, 160)

# ==========================================
# PLANT DISEASE INFORMATION DATABASE
# ==========================================

DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "type": "Bacterial Disease",
        "about": "Bacterial spot can affect pepper leaves, stems and fruits.",
        "symptoms": [
            "Small brown circular spots on leaves.",
            "Dark spots may develop on stems and fruit.",
            "Severe infections can cause leaf damage."
        ],
        "prevention": [
            "Use healthy seeds and disease-free transplants.",
            "Avoid overhead watering where practical.",
            "Keep tools clean and remove infected plant debris."
        ]
    },

    "Pepper__bell___healthy": {
        "type": "Healthy Plant",
        "about": "The model classified this pepper leaf as healthy.",
        "symptoms": [
            "No disease pattern was identified by the model.",
            "The leaf may have a generally healthy appearance."
        ],
        "prevention": [
            "Maintain suitable watering and sunlight.",
            "Inspect leaves regularly for unusual spots.",
            "Use healthy planting material."
        ]
    },

    "Potato___Early_blight": {
        "type": "Fungal Disease",
        "about": "Early blight commonly affects potato leaves and stems.",
        "symptoms": [
            "Round brown leaf spots.",
            "Larger spots may show concentric target-like rings.",
            "Yellowing may develop around affected areas."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Keep foliage dry where practical.",
            "Improve airflow and remove affected plant debris."
        ]
    },

    "Potato___Late_blight": {
        "type": "Oomycete Disease",
        "about": "Late blight is a serious potato disease that can affect foliage and tubers.",
        "symptoms": [
            "Large irregular brown or water-soaked leaf lesions.",
            "Pale tissue may surround affected areas.",
            "White growth may appear underneath leaves in humid conditions."
        ],
        "prevention": [
            "Use healthy seed potatoes.",
            "Monitor crops frequently during cool, wet weather.",
            "Remove infected crop material appropriately."
        ]
    },

    "Potato___healthy": {
        "type": "Healthy Plant",
        "about": "The model classified this potato leaf as healthy.",
        "symptoms": [
            "No disease pattern was identified by the model.",
            "Continue checking leaves and stems for changes."
        ],
        "prevention": [
            "Use healthy seed potatoes.",
            "Maintain suitable soil moisture.",
            "Inspect plants regularly."
        ]
    },

    "Tomato_Bacterial_spot": {
        "type": "Bacterial Disease",
        "about": "Bacterial spot affects tomato leaves, stems and fruits.",
        "symptoms": [
            "Small brown or dark circular leaf spots.",
            "Yellow halos may surround leaf spots.",
            "Fruit may develop raised, scabby spots."
        ],
        "prevention": [
            "Use healthy seeds and transplants.",
            "Avoid handling plants when foliage is wet.",
            "Clean gardening tools and remove crop debris."
        ]
    },

    "Tomato_Early_blight": {
        "type": "Fungal Disease",
        "about": "Early blight is a common tomato disease affecting leaves, stems and fruits.",
        "symptoms": [
            "Dark irregular leaf spots.",
            "Concentric target-like rings may develop.",
            "Yellowing and leaf loss may occur."
        ],
        "prevention": [
            "Water near the base of plants.",
            "Improve spacing and airflow.",
            "Remove affected leaves and plant debris."
        ]
    },

    "Tomato_Late_blight": {
        "type": "Oomycete Disease",
        "about": "Late blight can spread rapidly in favorable cool and wet conditions.",
        "symptoms": [
            "Water-soaked leaf lesions.",
            "Brown or dark affected areas.",
            "White growth may appear on infected tissue in humid conditions."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Monitor plants during cool, wet weather.",
            "Avoid prolonged leaf wetness where practical."
        ]
    },

    "Tomato_Leaf_Mold": {
        "type": "Fungal Disease",
        "about": "Tomato leaf mold is associated with humid growing conditions and commonly affects foliage.",
        "symptoms": [
            "Yellowish spots on the upper leaf surface.",
            "Olive-green or brown mold underneath leaves.",
            "Affected leaves may wither and dry."
        ],
        "prevention": [
            "Improve ventilation around plants.",
            "Reduce excessive humidity where possible.",
            "Remove affected plant debris."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "type": "Fungal Disease",
        "about": "Septoria leaf spot is a fungal disease that commonly affects tomato foliage.",
        "symptoms": [
            "Small circular leaf spots.",
            "Dark margins with tan or grey centers.",
            "Tiny dark dots may appear within spots."
        ],
        "prevention": [
            "Avoid overhead watering.",
            "Keep good airflow around plants.",
            "Remove affected leaves and crop debris."
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "type": "Pest Damage",
        "about": "Two-spotted spider mites are tiny pests that feed on plant sap.",
        "symptoms": [
            "Tiny pale or yellow spots on leaves.",
            "Leaves may become bronze or discolored.",
            "Fine webbing may appear during heavy infestations."
        ],
        "prevention": [
            "Inspect the undersides of leaves regularly.",
            "Maintain suitable plant watering.",
            "Monitor plants for early signs of infestation."
        ]
    },

    "Tomato__Target_Spot": {
        "type": "Fungal Disease",
        "about": "Target spot is a leaf disease associated with circular or irregular lesions.",
        "symptoms": [
            "Brown or dark leaf spots.",
            "Some lesions may show ring-like patterns.",
            "Affected spots may enlarge over time."
        ],
        "prevention": [
            "Avoid prolonged leaf wetness.",
            "Maintain adequate plant spacing.",
            "Remove affected plant debris."
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "type": "Viral Disease",
        "about": "Tomato yellow leaf curl is a viral disease associated with leaf yellowing and curling.",
        "symptoms": [
            "Yellowing of leaves.",
            "Upward curling or distortion of leaves.",
            "Stunted plant growth may occur."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Monitor plants for whitefly activity.",
            "Remove and manage infected plants according to local agricultural guidance."
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "type": "Viral Disease",
        "about": "Tomato mosaic virus can affect leaf appearance and plant growth.",
        "symptoms": [
            "Light and dark green mosaic patterns.",
            "Leaves may become curled or distorted.",
            "Plant growth may be reduced."
        ],
        "prevention": [
            "Use healthy seeds and planting material.",
            "Clean tools and hands after handling plants.",
            "Avoid moving plant sap between healthy and suspected infected plants."
        ]
    },

    "Tomato_healthy": {
        "type": "Healthy Plant",
        "about": "The model classified this tomato leaf as healthy.",
        "symptoms": [
            "No disease pattern was identified by the model.",
            "Continue monitoring leaves and plant growth."
        ],
        "prevention": [
            "Maintain suitable watering and nutrition.",
            "Inspect plants regularly.",
            "Keep the growing area clean."
        ]
    }
}

# ==========================================
# CUSTOM CSS - PREMIUM UI
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, rgba(34,197,94,0.12), transparent 35%),
        radial-gradient(ellipse at 90% 80%, rgba(20,184,166,0.12), transparent 35%),
        linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 45%, #f0fdfa 100%);
    color: #153b2a;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #063b2a, #08734b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 12px;
    border-radius: 10px;
}

/* Animated hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(120deg, #064e3b, #087f5b, #10b981);
    background-size: 200% 200%;
    animation: gradientMove 8s ease infinite, fadeIn 1s ease;
    color: white;
    box-shadow: 0 15px 35px rgba(6,78,59,0.20);
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-bottom: 10px;
}

.hero p {
    color: #dcfce7 !important;
    font-size: 17px;
}

.hero-tag {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(255,255,255,0.16);
    color: white;
    font-size: 12px;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

/* Floating decorative circle */
.hero::after {
    content: "🌿";
    position: absolute;
    right: 7%;
    top: 15%;
    font-size: 115px;
    opacity: 0.18;
    animation: float 4s ease-in-out infinite;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #bbf7d0;
    padding: 18px 12px;
    min-height: 155px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(15,70,45,0.08);
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeUp 0.8s ease both;
}
.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 30px rgba(16,185,129,0.18);
}

.metric-icon {
    font-size: 30px;
}

.metric-value {
    font-size: 31px;
    font-weight: 800;
    color: #087443;
    margin: 8px 0;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
}

/* Section headings */
.section-heading {
    font-size: 24px;
    font-weight: 800;
    color: #075c3b;
    margin: 30px 0 15px;
}

/* Feature cards */
.feature-card {
    height: 175px;
    padding: 22px;
    background: rgba(255,255,255,0.88);
    border-radius: 18px;
    border: 1px solid #d1fae5;
    box-shadow: 0 5px 20px rgba(0,0,0,0.04);
    transition: all 0.3s ease;
    animation: fadeUp 1s ease both;
}

.feature-card:hover {
    transform: translateY(-6px);
    border-color: #10b981;
    box-shadow: 0 12px 28px rgba(16,185,129,0.15);
}

.feature-icon {
    font-size: 32px;
}

.feature-card h3 {
    color: #075c3b !important;
    font-size: 18px;
}

.feature-card p {
    color: #64748b !important;
    font-size: 14px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #064e3b, #087f5b);
    color: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(6,78,59,0.18);
    animation: fadeUp 0.7s ease;
}

.result-card h2,
.result-card h3,
.result-card p {
    color: white !important;
}

/* Upload section */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.75);
    border: 2px dashed #10b981;
    border-radius: 18px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #087443, #10b981);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 11px 22px;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 7px 18px rgba(16,185,129,0.28);
}

/* Images */
img {
    border-radius: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 22px;
    margin-top: 35px;
    border-top: 1px solid #d1fae5;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes float {
    0%, 100% {transform: translateY(0);}
    50% {transform: translateY(-16px);}
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Mobile adjustments */
@media (max-width: 700px) {
    .hero {padding: 25px;}
    .hero h1 {font-size: 30px !important;}
    .hero::after {font-size: 65px; right: 3%;}
}

</style>
""", unsafe_allow_html=True)

# Initialize SQLite database
create_table()
# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = [
        {
            "Date": record[1],
            "Image": record[2],
            "Prediction": record[3],
            "Confidence": record[4]
        }
        for record in get_all_predictions()
    ]

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_disease(image):

    image = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if image_array.shape != (1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(f"Unexpected image array shape: {image_array.shape}")

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    disease = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = [
        {
            "Disease": class_names[int(i)].replace("_", " "),
            "Confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top_indices
    ]

    return disease, confidence, top_predictions


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🌿 PlantAI")
    st.caption("SMART AGRICULTURE • AI")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Disease Detection",
            "Prediction History",
            "About Project"
        ],
        key="main_menu"
    )

    st.divider()

    st.markdown("### 🤖 AI Model")
    st.success("Model loaded successfully")

    st.markdown("### 📊 Model Statistics")
    st.write(f"Classification: {len(class_names)} classes")
    st.write("Validation accuracy: 86.65%")

    st.divider()
    st.caption("PlantAI | Deep Learning Project")


# ==========================================
# HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI POWERED AGRICULTURE</div>
    <h1>🌿 PlantAI</h1>
    <p>Intelligent Plant Disease Detection System</p>
    <p>Helping identify plant leaf diseases with deep learning.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
    '<div class="section-heading">Your Plant Health Overview 🌱</div>',
    unsafe_allow_html=True
)

    st.write(
        "Analyze plant leaf images and explore AI-powered disease classification."
    )

    
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🌱</div>
            <div class="metric-value">{class_count}</div>
            <div class="metric-label">Disease Classes</div>
        </div>
        """.format(class_count=len(class_names)), unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">86.65%</div>
            <div class="metric-label">Validation Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔬</div>
            <div class="metric-value">{len(st.session_state.history)}</div>
            <div class="metric-label">Session Predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Explore PlantAI</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <h3>Disease Detection</h3>
            <p>Upload a plant leaf image and analyze it using AI.</p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Prediction Analysis</h3>
            <p>View predicted disease categories and confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h3>Prediction History</h3>
            <p>Review and download your previous session results.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">How PlantAI Works</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.info("📤 **Step 1: Upload**\n\nChoose a clear plant leaf image.")

    with b:
        st.info("🧠 **Step 2: Analyze**\n\nThe deep learning model analyzes the image.")

    with c:
        st.info("📈 **Step 3: Results**\n\nView the predicted class and confidence.")

    st.button(
        "Start Disease Detection →",
        on_click=lambda: st.session_state.update(
            {"main_menu": "Disease Detection"}
        )
    )


# ==========================================
# DISEASE DETECTION
# ==========================================

elif page == "Disease Detection":

    st.markdown(
        '<div class="section-heading">🔬 AI Plant Disease Scanner</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a clear photograph of a plant leaf to receive an AI-generated classification."
    )

    uploaded_file = st.file_uploader(
        "Choose your plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="Uploaded plant leaf",
                use_container_width=True
            )

        with right:
            st.markdown("### Image Information")
            st.write("**Filename:**", uploaded_file.name)
            st.write("**Image dimensions:**", image.size)
            st.write("**Format:**", uploaded_file.type)

            st.success("Image uploaded successfully!")

        if st.button(
            "🚀 Analyze Plant Leaf",
            use_container_width=True
        ):

            with st.spinner("AI is analyzing your plant image..."):

                disease, confidence, top_predictions = predict_disease(image)

                result = {
                    "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Image": uploaded_file.name,
                    "Prediction": disease.replace("_", " "),
                    "Confidence": round(confidence, 2)
                }

                save_prediction(
                    date=result["Date"],
                    image_name=result["Image"],
                    disease=result["Prediction"],
                    confidence=result["Confidence"]
                )

                st.session_state.history.append(result)
                st.session_state.last_result = result
                st.session_state.last_top_predictions = top_predictions

        if st.session_state.last_result:

            result = st.session_state.last_result
            top_predictions = st.session_state.get(
                "last_top_predictions", []
            )

            st.markdown("### 🎯 AI Analysis Result")

            st.markdown(f"""
            <div class="result-card">
                <p>DETECTED PLANT CONDITION</p>
                <h2>{result["Prediction"]}</h2>
                <h1>{result["Confidence"]:.2f}%</h1>
                <p>Model confidence score</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(int(result["Confidence"]), 100)
            )

            if result["Confidence"] < 60:
                st.warning(
                    "Low model confidence. Try another clear image and seek expert confirmation."
                )
            else:
                st.info(
                    "The model has produced a classification. This is not a confirmed field diagnosis."
                )

            # ==========================================
            # AUTOMATIC DISEASE INFORMATION
            # ==========================================

            disease_info = DISEASE_INFO.get(disease, {
                "type": "Unknown",
                "about": "No additional information is available.",
                "symptoms": [],
                "prevention": []
            })

            st.markdown("### 🌿 Plant Health Information")

            st.markdown(f"""
            <div class="feature-card">
                <h3>{disease.replace('_', ' ')}</h3>
                <p><b>Category:</b> {disease_info["type"]}</p>
                <p>{disease_info["about"]}</p>
            </div>
            """, unsafe_allow_html=True)

            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.markdown("#### 🔎 Common Symptoms")

                for symptom in disease_info["symptoms"]:
                    st.write("•", symptom)

            with info_col2:
                st.markdown("#### 🌱 General Prevention")

                for advice in disease_info["prevention"]:
                    st.write("•", advice)

            st.warning(
                "This information is for educational purposes. "
                "The AI prediction is not a confirmed diagnosis. "
                "Consult an agricultural expert before deciding on treatment."
            )

            st.markdown("### 📊 Top 3 Model Predictions")

            top_df = pd.DataFrame(top_predictions)

            top_df["Confidence"] = top_df["Confidence"].map(
                lambda x: f"{x:.2f}%"
            )

            st.dataframe(
                top_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Predictions are limited to the 15 classes in the training dataset. "
                "Consult an agricultural expert before making treatment decisions."
            )

    else:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🍃</div>
            <h3>Ready to analyze your plant?</h3>
            <p>Upload a JPG, JPEG or PNG leaf photograph above to begin.</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PREDICTION HISTORY
# ==========================================


elif page == "Prediction History":

    st.markdown(
        '<div class="section-heading">📚 Permanent Prediction History</div>',
        unsafe_allow_html=True
    )

    records = get_all_predictions()

    if records:

        history_df = pd.DataFrame(
            records,
            columns=[
                "ID",
                "Date",
                "Image Name",
                "Disease",
                "Confidence"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Predictions", len(history_df))

        with col2:
            st.metric(
                "Average Confidence",
                f'{history_df["Confidence"].mean():.2f}%'
            )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv,
            file_name="plantai_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear All History"):

            clear_predictions()

            st.session_state.history = []
            st.session_state.last_result = None

            st.success("Prediction history cleared!")
            st.rerun()

    else:
        st.info(
            "No prediction records found. "
            "Upload a plant image to create your first record."
        )
# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.markdown(
        '<div class="section-heading">🌿 About PlantAI</div>',
        unsafe_allow_html=True
    )

    st.write("""
    PlantAI is an AI-powered plant disease classification project.

    The system uses a MobileNetV2-based deep learning model
    to classify uploaded plant leaf images into 15 categories.
    """)

    st.markdown("### ⚙️ Technology Stack")

    technologies = [
        ("🐍", "Python"),
        ("🧠", "TensorFlow and Keras"),
        ("📷", "Pillow"),
        ("🔢", "NumPy"),
        ("📊", "Pandas"),
        ("🌐", "Streamlit"),
        ("🤖", "MobileNetV2")
    ]

    for icon, tech in technologies:
        st.write(f"{icon}  {tech}")

    st.markdown("### 📈 Model Performance")

    st.metric("Validation Accuracy", "86.65%")
    st.metric("Number of Classes", "15")

    st.markdown("### 🎯 Project Objective")

    st.write("""
    To demonstrate how deep learning and computer vision
    can support the early identification of plant leaf conditions.
    """)

    st.warning(
        "PlantAI is an educational classification tool. "
        "It does not replace laboratory testing or professional agricultural advice."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🌿 <b>PlantAI</b> | Smart Agriculture with Artificial Intelligence
    <br><br>
    Built with Python • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)
