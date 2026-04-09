# ============================================================
#  FACIAL EXPRESSION DETECTOR – Beautiful UI + Fast Webcam
# ============================================================

import os, cv2, torch, numpy as np, streamlit as st, torchvision, time
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Expression Detector | AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# BEAUTIFUL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Hero section */
.hero {
    text-align: center;
    padding: 40px 0 20px 0;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 8px;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 24px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #64748b;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* Upload zone */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(102,126,234,0.4);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(102,126,234,0.8);
    background: rgba(102,126,234,0.05);
}

/* Expression result card */
.expr-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s;
}
.expr-card:hover { transform: translateY(-2px); }

.expr-emoji { font-size: 2.8rem; margin-bottom: 8px; }
.expr-label {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.expr-conf {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 12px;
}

/* Probability bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
}
.prob-label { width: 68px; font-size: 0.72rem; color: #64748b; text-align: right; }
.prob-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 999px; }
.prob-pct { width: 32px; font-size: 0.72rem; color: #94a3b8; text-align: right; }

/* Metric cards */
.metric-grid { display: flex; gap: 12px; margin-top: 20px; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.metric-num {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

/* Section title */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
}

/* Tip box */
.tip-box {
    background: rgba(102,126,234,0.08);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 16px;
}
.tip-title { color: #818cf8; font-weight: 700; font-size: 0.9rem; margin-bottom: 10px; }
.tip-item { color: #64748b; font-size: 0.85rem; margin: 5px 0; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border: none !important;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(102,126,234,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(102,126,234,0.5);
}

/* Success/error messages */
.stSuccess { background: rgba(16,185,129,0.1) !important; border-color: rgba(16,185,129,0.3) !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border-color: rgba(239,68,68,0.3)  !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/omer-khan/Facial-expression-detector/resolve/main/my_model.pth"
MODEL_PATH = "my_model.pth"

EXPRESSION_CLASSES = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

EXPRESSION_EMOJI = {
    "Angry":"😡","Disgust":"🤢","Fear":"😨",
    "Happy":"😄","Neutral":"😐","Sad":"😢","Surprise":"😲"
}
EXPRESSION_COLOR = {
    "Angry":"#ef4444","Disgust":"#22c55e","Fear":"#a855f7",
    "Happy":"#f59e0b","Neutral":"#94a3b8","Sad":"#3b82f6","Surprise":"#f97316"
}
COLOUR_BGR = {
    "Angry":(0,0,220),"Disgust":(0,180,0),"Fear":(180,0,180),
    "Happy":(0,200,0),"Neutral":(160,160,160),"Sad":(220,80,0),"Surprise":(0,160,255)
}

IMG_SIZE            = 260
FACE_CONF_THRESHOLD = 0.80
MIN_FACE_SIZE       = 20
UPSCALE_FACTOR      = 2.0


# ─────────────────────────────────────────────
# LOAD MODEL & DETECTOR
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model from Hugging Face if not found locally
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Hugging Face (34MB) — only happens once..."):
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    m = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    m.to(device); m.eval()
    return m, device

@st.cache_resource
def load_detector():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(keep_all=True, device=device, post_process=False)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # match training (black & white dataset)
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def detect_faces(frame_rgb, detector, upscale=UPSCALE_FACTOR):
    h, w = frame_rgb.shape[:2]
    big  = cv2.resize(frame_rgb,(int(w*upscale),int(h*upscale))) if upscale!=1.0 else frame_rgb
    boxes, probs = detector.detect(Image.fromarray(big))
    results = []
    if boxes is None:
        return results
    for box, prob in zip(boxes, probs):
        if prob is None or prob < FACE_CONF_THRESHOLD:
            continue
        x1,y1,x2,y2 = box
        x1=int(max(0,x1)/upscale); y1=int(max(0,y1)/upscale)
        x2=int(x2/upscale);        y2=int(y2/upscale)
        x1,y1=max(0,x1),max(0,y1)
        x2,y2=min(w,x2),min(h,y2)
        if (x2-x1)>=MIN_FACE_SIZE and (y2-y1)>=MIN_FACE_SIZE:
            results.append((x1,y1,x2,y2))
    return results

def predict_expression(crop_rgb, model, device):
    t = preprocess(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    idx = np.argmax(p)
    return EXPRESSION_CLASSES[idx], float(p[idx]), p

def draw_box(frame_bgr, x1, y1, x2, y2, label, conf):
    """Sharp, large, readable box — works on any image size."""
    colour = COLOUR_BGR.get(label, (255,255,255))
    # thick border box
    cv2.rectangle(frame_bgr,(x1,y1),(x2,y2), colour, 3)
    # large readable text
    text  = f"{label}  {conf*100:.0f}%"
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.9
    thick = 2
    (tw,th),_ = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    bx1, by1 = x1, max(0, y1 - th - pad*2)
    bx2, by2 = x1 + tw + pad*2, y1
    # solid filled background for text (no blur)
    cv2.rectangle(frame_bgr,(bx1,by1),(bx2,by2), colour, -1)
    cv2.putText(frame_bgr, text, (bx1+pad, by2-pad),
                font, scale, (0,0,0), thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 Expression Detector</div>
    <div class="hero-sub">AI-Powered Facial Emotion Recognition • EfficientNet-B2 + MTCNN</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
with st.spinner("Initializing AI model..."):
    try:
        model, device = load_model()
        detector      = load_detector()
        st.markdown(f"""
        <div style="text-align:center">
            <span class="status-badge">● Model ready on {device.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.info(f"Make sure **{MODEL_PATH}** is in the same folder as app.py")
        st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸   Image Upload", "📷   Live Webcam"])


# ══════════════════════════════════════════════
# TAB 1 – IMAGE UPLOAD
# ══════════════════════════════════════════════
with tab1:

    # Upload row
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload", type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ── Standard square display size (no blur — pad with black, don't stretch) ──
        SQ = 400   # square canvas size in pixels
        def make_square(bgr_img, size=SQ):
            """Fit image into a square canvas with black padding — sharp, no stretch."""
            h, w = bgr_img.shape[:2]
            scale = min(size/w, size/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas  = np.zeros((size, size, 3), dtype=np.uint8)
            x_off   = (size - new_w) // 2
            y_off   = (size - new_h) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            return canvas, scale, x_off, y_off

        with st.spinner("Analyzing faces..."):
            boxes = detect_faces(img_rgb, detector)

        all_results = []
        result_bgr  = img_bgr.copy()

        if boxes:
            for x1,y1,x2,y2 in boxes:
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0: continue
                label, conf, probs = predict_expression(crop, model, device)
                draw_box(result_bgr, x1, y1, x2, y2, label, conf)
                all_results.append((label, conf, probs))

        # Make square versions for display (sharp, consistent size)
        orig_sq,  _, _, _ = make_square(img_bgr)
        result_sq, _, _, _ = make_square(result_bgr)

        orig_disp   = cv2.cvtColor(orig_sq,   cv2.COLOR_BGR2RGB)
        result_disp = cv2.cvtColor(result_sq,  cv2.COLOR_BGR2RGB)

        # Side by side square images
        col_orig, col_res = st.columns(2, gap="large")
        with col_orig:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🖼 Original</div>', unsafe_allow_html=True)
            st.image(orig_disp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Detection Result</div>', unsafe_allow_html=True)
            if not boxes:
                st.warning("⚠️ No faces detected. Try a clearer or closer image.")
            else:
                st.image(result_disp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Expression cards + metrics below images ──
        if boxes and all_results:
            st.markdown('<div class="section-title" style="text-align:center;margin-top:8px">🎭 Expression Analysis</div>', unsafe_allow_html=True)

            # Summary metrics row
            avg_conf = np.mean([c for _,c,_ in all_results])
            dominant = max(all_results, key=lambda x: x[1])
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-num">{len(boxes)}</div>
                    <div class="metric-lbl">Faces Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-num">{avg_conf*100:.0f}%</div>
                    <div class="metric-lbl">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-num">{EXPRESSION_EMOJI.get(dominant[0],'')}</div>
                    <div class="metric-lbl">Dominant Emotion</div>
                </div>
                <div class="metric-card">
                    <div class="metric-num">{dominant[0]}</div>
                    <div class="metric-lbl">Top Expression</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Face cards — max 4 per row
            cols = st.columns(min(len(all_results), 4))
            for i, (col, (label, conf, probs)) in enumerate(zip(cols, all_results)):
                emoji = EXPRESSION_EMOJI.get(label,"")
                color = EXPRESSION_COLOR.get(label,"#94a3b8")
                bars_html = ""
                for cls, p in zip(EXPRESSION_CLASSES, probs):
                    c = EXPRESSION_COLOR.get(cls,"#94a3b8")
                    bars_html += f"""
                    <div class="prob-row">
                        <div class="prob-label">{cls}</div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{p*100:.0f}%;background:{c}"></div>
                        </div>
                        <div class="prob-pct">{p*100:.0f}%</div>
                    </div>"""
                with col:
                    st.markdown(f"""
                    <div class="expr-card">
                        <div class="expr-emoji">{emoji}</div>
                        <div class="expr-label" style="color:{color}">Face #{i+1}</div>
                        <div class="expr-label" style="color:{color}">{label}</div>
                        <div class="expr-conf" style="color:{color}">{conf*100:.1f}%</div>
                        <div style="margin-top:10px">{bars_html}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 – LIVE WEBCAM (optimized for speed)
# ══════════════════════════════════════════════
with tab2:

    # ── Warning banner ──
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(239,68,68,0.08));
        border: 1px solid rgba(245,158,11,0.35);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    ">
        <span style="font-size:1.6rem">⚠️</span>
        <div>
            <div style="color:#f59e0b; font-weight:700; font-size:0.95rem; margin-bottom:6px">
                Webcam is best used locally on your PC
            </div>
            <div style="color:#94a3b8; font-size:0.83rem; line-height:1.6">
                When accessed via a cloud link, the live webcam may feel slow or laggy
                because every video frame travels over the internet to the server and back.<br>
                <strong style="color:#fbbf24">For best experience → run this app on your own laptop:</strong>
                <code style="background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:6px; color:#a78bfa; font-size:0.8rem">
                streamlit run app.py
                </code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cam, col_info = st.columns([1.6, 1], gap="large")

    with col_cam:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Live Camera Feed</div>', unsafe_allow_html=True)

        class ExpressionProcessor(VideoProcessorBase):
            def __init__(self):
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.detector  = MTCNN(keep_all=True, device=dev, post_process=False)
                self.model     = model
                self.device    = device
                self.frame_cnt = 0
                self.last_boxes = []
                self.last_preds = []

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.frame_cnt += 1

                # ⚡ Only detect faces every 3 frames (speeds up webcam)
                if self.frame_cnt % 3 == 0:
                    try:
                        h, w = img_rgb.shape[:2]
                        pil  = Image.fromarray(img_rgb)
                        boxes_raw, probs_raw = self.detector.detect(pil)
                        self.last_boxes = []
                        self.last_preds = []

                        if boxes_raw is not None:
                            for box, prob in zip(boxes_raw, probs_raw):
                                if prob is None or prob < FACE_CONF_THRESHOLD:
                                    continue
                                x1,y1,x2,y2 = [int(v) for v in box]
                                x1,y1=max(0,x1),max(0,y1)
                                x2,y2=min(w,x2),min(h,y2)
                                if (x2-x1)<MIN_FACE_SIZE or (y2-y1)<MIN_FACE_SIZE:
                                    continue
                                crop = img_rgb[y1:y2,x1:x2]
                                if crop.size==0: continue
                                label,conf,_ = predict_expression(crop,self.model,self.device)
                                self.last_boxes.append((x1,y1,x2,y2))
                                self.last_preds.append((label,conf))
                    except Exception:
                        pass

                # Draw last known boxes on every frame (smooth appearance)
                for (x1,y1,x2,y2),(label,conf) in zip(self.last_boxes, self.last_preds):
                    draw_box(img_bgr, x1, y1, x2, y2, label, conf)

                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        webrtc_streamer(
            key="expr-live",
            video_processor_factory=ExpressionProcessor,
            rtc_configuration=RTCConfiguration({
                "iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]
            }),
            media_stream_constraints={"video":{"width":640,"height":480},"audio":False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Expressions Guide</div>', unsafe_allow_html=True)

        for cls in EXPRESSION_CLASSES:
            color = EXPRESSION_COLOR.get(cls,"#94a3b8")
            emoji = EXPRESSION_EMOJI.get(cls,"")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:10px;
                        background:rgba(255,255,255,0.03);border-radius:10px;margin:6px 0;
                        border:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:1.5rem">{emoji}</span>
                <span style="color:{color};font-weight:700;font-size:0.95rem">{cls}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box">
            <div class="tip-title">💡 Tips for best results</div>
            <div class="tip-item">• Face the camera directly</div>
            <div class="tip-item">• Good front lighting</div>
            <div class="tip-item">• Stay within 1 meter</div>
            <div class="tip-item">• Avoid strong backlight</div>
            <div class="tip-item">• Exaggerate your expression</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px 0;color:#334155;font-size:0.8rem">
    Built with EfficientNet-B2 + MTCNN + Streamlit &nbsp;•&nbsp; UET Peshawar Internship Project
</div>
""", unsafe_allow_html=True)
