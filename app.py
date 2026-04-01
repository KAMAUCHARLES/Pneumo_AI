import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ============================================
# Page config - wide layout with custom theme
# ============================================
st.set_page_config(
    page_title="PneumoAI | Precision Pneumonia Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for professional medical UI
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #F5F7FA 0%, #EEF2F5 100%);
    }
    
    /* Main container */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0F3B5C 0%, #1A6F9F 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .main-header p {
        font-size: 1rem;
        color: #5F6C80;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Result cards with accent borders */
    .result-card {
        border-radius: 20px;
        padding: 1.25rem;
        margin: 1rem 0;
        transition: all 0.2s;
    }
    
    .result-bacterial {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #C62828;
    }
    
    .result-viral {
        background: linear-gradient(135deg, #FFF8E7 0%, #FFECB3 100%);
        border-left: 6px solid #F9A825;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #2E7D32;
    }
    
    /* Confidence bar animation */
    .confidence-bar {
        background: #E0E7FF;
        border-radius: 20px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 0.75rem 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #1A6F9F, #4A90E2);
        border-radius: 20px;
        height: 100%;
        width: 0%;
        transition: width 0.6s cubic-bezier(0.2, 0.9, 0.4, 1.1);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #B0C4DE;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.5);
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #1A6F9F;
        background: rgba(255, 255, 255, 0.8);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A6F9F;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6C757D;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #0F3B5C 0%, #1A6F9F 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F1F1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Header section
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🫁 PneumoAI</h1>
    <p>AI‑Powered Pneumonia Diagnosis & Subtyping | Bacterial vs Viral</p>
    <p style="font-size: 0.85rem; color: #5F6C80;">Designed for Kenyan County Hospitals</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Sidebar - Clinical context and metrics
# ============================================
with st.sidebar:
    st.markdown("## 🏥 Clinical Intelligence")
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="metric-value">90.5%</p><p class="metric-label">Bacterial<br>Recall</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="metric-value">85.1%</p><p class="metric-label">Viral<br>Recall</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="metric-value">80.5%</p><p class="metric-label">Overall<br>Accuracy</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Why Distinguish?")
    st.markdown("""
    | Finding | Treatment |
    |---------|-----------|
    | **Bacterial** | 💊 Antibiotics required |
    | **Viral** | 🩺 Supportive care only |
    | **Wrong treatment** | ⚠️ Antibiotic resistance risk |
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛡️ Ethical Safeguards")
    st.markdown("""
    - **Human‑in‑the‑loop** – final decision by clinician
    - **Explainable AI** – Grad‑CAM heatmaps
    - **Offline** – runs on hospital laptops
    - **Privacy** – no data leaves device
    """)
    
    st.markdown("---")
    
    st.markdown("### 🇰🇪 Kenya Context")
    st.markdown("Pneumonia is the **2nd leading cause of child death** in Kenya. Our tool reduces diagnostic delays and guides appropriate treatment.")

# ============================================
# Load model (cached)
# ============================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 3)
    model.load_state_dict(torch.load('pneumonia_multiclass.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# ============================================
# Preprocessing
# ============================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ============================================
# Grad-CAM heatmap generator
# ============================================
def generate_heatmap(model, input_tensor, target_class, device):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    img_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# ============================================
# Main content
# ============================================
with st.spinner("Loading AI model..."):
    model, device = load_model()

CLASSES = ['BACTERIAL', 'NORMAL', 'VIRAL']

# Upload area with custom styling
st.markdown("""
<div class="upload-area">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">📤 Drag & Drop or Click to Upload</p>
    <p style="font-size: 0.8rem; color: #6C757D;">Supports JPG, JPEG, PNG | Chest X-ray images only</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Two-column layout
    col_img, col_res = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption("Uploaded chest X-ray")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_res:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probs, 1)
                    confidence = confidence.item()
                    pred_class = CLASSES[pred.item()]
                
                # Result card with confidence bar
                if pred_class == 'BACTERIAL':
                    card_class = "result-bacterial"
                    title = "⚠️ Bacterial Pneumonia Detected"
                    advice = "🔴 Prescribe appropriate antibiotics. Monitor oxygen saturation."
                elif pred_class == 'VIRAL':
                    card_class = "result-viral"
                    title = "🦠 Viral Pneumonia Detected"
                    advice = "🟡 Supportive care. Antibiotics not indicated."
                else:
                    card_class = "result-normal"
                    title = "✅ Normal"
                    advice = "🟢 No pneumonia detected. Continue clinical assessment."
                
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h3 style="margin-top: 0;">{title}</h3>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                    </div>
                    <p style="margin-top: 0.75rem;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Heatmap section
                st.markdown("### 🔥 Explainability")
                st.caption("The highlighted areas show where the model focused")
                try:
                    heatmap = generate_heatmap(model, input_tensor, CLASSES.index(pred_class), device)
                    st.image(heatmap, use_container_width=True)
                except Exception as e:
                    st.warning("Heatmap generation temporarily unavailable")
    
    # Clinical guidance footer
    st.markdown("---")
    st.info("📖 **Clinical note:** Correlate AI findings with patient history, symptoms, and physical examination. This is a second opinion tool.")

else:
    # Welcome placeholder
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 1rem;">
        <p style="font-size: 1rem;">🩻 Ready for analysis</p>
        <p style="font-size: 0.85rem; color: #6C757D;">Upload a chest X-ray to begin</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #E0E0E0; color: #6C757D; font-size: 0.75rem;">
    <p>PneumoAI | ResNet18 + Grad‑CAM | Multi‑class (Bacterial / Viral / Normal)</p>
    <p>Designed for Kenyan County Hospitals | AI in Health Hackathon 2026</p>
</div>
""", unsafe_allow_html=True)
