import streamlit as st
import requests
from PIL import Image
import io
import base64

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Plant Health Diagnosis System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Base64 Background Image ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_image("/Users/jessekariuki/Downloads/isolated-green-monstera-leaf-transparent-background-free-png.png")

# ---------- Custom CSS ----------
st.markdown(f"""
    <style>
    /* Background color + image */
    .stApp {{
        background-color: #CBFBB5;
        background-image: url("data:image/png;base64,{bg_image}");
        background-repeat: no-repeat;
        background-position: bottom right;
        background-size: 420px;
        background-attachment: fixed;
    }}
    /* Ensure all text uses sans-serif */
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif !important;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 0rem;
        background-color: transparent;
    }}
    .main-title {{
        font-size: 4.2rem;
        color: #2E7D32;
        font-weight: 700;
        text-align: center;
        margin-top: 3rem;
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif !important;
        letter-spacing: -0.02em;
    }}
    .sub-title {{
        font-size: 1.6rem;
        color: #333;
        text-align: center;
        margin-top: 0.5rem;
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif !important;
        font-weight: 300;
    }}
    .diagnosis-box {{
        background-color: #E8F5E9;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 5px solid #2E7D32;
    }}
    .disclaimer-box {{
        background-color: #FFEBEE;
        border-left: 4px solid #D32F2F;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin-top: 1.5rem;
    }}
    .confidence-high {{
        color: #2E7D32;
        font-weight: 600;
    }}
    .confidence-low {{
        color: #C62828;
        font-weight: 600;
    }}
    .about-box {{
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }}
    /* BUTTON STYLES */
    .stButton button {{
        background: linear-gradient(90deg, rgba(42, 123, 155, 1) 0%, rgba(7, 217, 94, 1) 42%, rgba(237, 221, 83, 1) 100%) !important;
        color: #000000 !important;
        text-shadow: none !important;
        border: none !important;
        padding: 1.5rem 2.5rem !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        min-height: 80px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }}
    .stButton button:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
        filter: brightness(1.05) !important;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        font-family: 'Inter', 'Helvetica', 'Arial', sans-serif !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- API URL ----------
API_URL = "http://localhost:8000/diagnose"

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "uploaded_image_display" not in st.session_state:
    st.session_state.uploaded_image_display = None

# ---------- Navigation Functions ----------
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

def reset_diagnosis():
    st.session_state.diagnosis_result = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_display = None
    navigate_to("home")

# ---------- PAGE: HOME ----------
def page_home():
    st.markdown('<p class="main-title">🌿 Let\'s Get Started</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Let\'s check the health of your plant</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌱 Get Started", use_container_width=True):
            navigate_to("upload")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📖 How It Works", use_container_width=True):
            navigate_to("about")

# ---------- PAGE: UPLOAD ----------
def page_upload():
    st.markdown("## 📤 Upload a Plant Photo")
    st.markdown("Choose a clear image of the affected leaf to get started.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_image_bytes = uploaded_file.getvalue()
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image_display = image
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if st.session_state.uploaded_image_display is not None:
            if st.button("🔍 Analyze Plant", type="primary", use_container_width=True):
                navigate_to("diagnosis")
        else:
            st.info("👆 Please upload an image first.")
    
    if st.button("← Back to Home", use_container_width=True):
        reset_diagnosis()

# ---------- PAGE: DIAGNOSIS ----------
def page_diagnosis():
    st.markdown("## 🔬 Analyzing your plant...")
    st.markdown("Please wait while the AI processes your image.")
    
    if st.session_state.uploaded_image_display is not None:
        st.image(st.session_state.uploaded_image_display, caption="Analyzing...", use_container_width=False, width=300)
    
    with st.spinner("The AI is examining the leaf for diseases..."):
        try:
            if st.session_state.uploaded_image_bytes is None:
                st.error("No image found. Please upload an image first.")
                if st.button("← Back to Upload"):
                    navigate_to("upload")
                return
            
            files = {"file": st.session_state.uploaded_image_bytes}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.diagnosis_result = result
            else:
                st.session_state.diagnosis_result = {
                    "success": False,
                    "message": f"Server error: {response.status_code}",
                    "detail": response.text
                }
        except requests.exceptions.ConnectionError:
            st.session_state.diagnosis_result = {
                "success": False,
                "message": "Cannot connect to the backend server. Make sure FastAPI is running on port 8000."
            }
        except Exception as e:
            st.session_state.diagnosis_result = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    navigate_to("results")

# ---------- PAGE: RESULTS ----------
def page_results():
    result = st.session_state.diagnosis_result
    
    if result is None:
        st.warning("No diagnosis result found. Please upload an image first.")
        if st.button("← Back to Upload"):
            navigate_to("upload")
        return
    
    st.markdown("## 📋 Diagnosis Results")
    
    if st.session_state.uploaded_image_display is not None:
        st.image(st.session_state.uploaded_image_display, caption="Uploaded Image", use_container_width=False, width=300)
    
    if result.get("success") and result.get("confidence", 0) >= 0.70:
        disease = result.get("disease", "Unknown")
        confidence = result.get("confidence", 0.0)
        treatment = result.get("treatment", "No treatment available.")
        disclaimer = result.get("disclaimer", "")
        message = result.get("message", "")
        
        st.markdown(f"""
        <div class="diagnosis-box">
            <h3 style="margin-top:0;">✅ Diagnosis Complete</h3>
            <p style="font-size:1.3rem; font-weight:600; margin:0.2rem 0;">Disease: {disease}</p>
            <p style="font-size:1.1rem; margin:0.2rem 0;">
                Confidence: <span class="confidence-high">{confidence:.2%}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💊 Treatment Plan")
        st.markdown(treatment)
        
        if disclaimer:
            st.markdown(f'<div class="disclaimer-box">⚠️ {disclaimer}</div>', unsafe_allow_html=True)
        
        if message and message != "Diagnosis completed successfully.":
            st.info(message)
    else:
        confidence = result.get("confidence")
        message = result.get("message", "Unable to diagnose the plant disease.")
        
        st.error(f"❌ {message}")
        
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2%}")
        
        disclaimer = result.get("disclaimer")
        if disclaimer:
            st.markdown(f'<div class="disclaimer-box">⚠️ {disclaimer}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Diagnosis", use_container_width=True):
            st.session_state.diagnosis_result = None
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_display = None
            navigate_to("upload")
    with col2:
        if st.button("🏠 Home", use_container_width=True):
            reset_diagnosis()

# ---------- PAGE: ABOUT ----------
def page_about():
    st.markdown("## 📖 About This System")
    
    st.markdown("""
    <div class="about-box">
        <h3>🌱 Plant Health Diagnosis & Recommendation</h3>
        <p>This AI-powered tool helps you identify plant diseases from leaf images and provides safe, actionable treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.write("Take a photo of the affected leaf and upload it.")
    with col2:
        st.markdown("**2. AI Analysis**")
        st.write("Our deep learning model analyzes the image and identifies the disease.")
    with col3:
        st.markdown("**3. Treatment**")
        st.write("You receive a detailed treatment plan based on verified agricultural sources.")
    
    st.markdown("### 🧪 Technology")
    st.markdown("""
    - **CNN Classifier:** EfficientNet-B0 trained on the PlantVillage dataset (38 disease classes, 88% accuracy).
    - **RAG System:** Retrieves treatment information from university extension guides (UC IPM, etc.).
    - **Safety First:** All advice includes a mandatory disclaimer and confidence threshold.
    """)
    
    st.markdown("### 👨‍🌾 Intended Use")
    st.markdown("""
    This system is designed for **educational and research purposes**. It is **not a substitute** for professional agronomic advice.
    Always verify treatments with a local expert before application.
    """)
    
    if st.button("← Back to Home", use_container_width=True):
        navigate_to("home")

# ---------- PAGE ROUTING ----------
def main():
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "upload":
        page_upload()
    elif st.session_state.page == "diagnosis":
        page_diagnosis()
    elif st.session_state.page == "results":
        page_results()
    elif st.session_state.page == "about":
        page_about()
    else:
        page_home()

if __name__ == "__main__":
    main()