import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import io
import os
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }
    .stApp {
        background: #0d0d0d;
        color: #e8e8e8;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .stButton > button {
        background: #00ff88;
        color: #0d0d0d;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #00cc6a;
        transform: translateY(-1px);
    }
    .stSlider > div {
        color: #e8e8e8;
    }
    .upload-box {
        border: 2px dashed #333;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #111;
    }
    .detection-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .detection-table th {
        background: #1a1a1a;
        color: #00ff88;
        font-family: 'Space Mono', monospace;
        padding: 0.5rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid #333;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .detection-table td {
        padding: 0.45rem 0.75rem;
        border-bottom: 1px solid #1f1f1f;
        color: #ccc;
    }
    .detection-table tr:hover td {
        background: #1a1a1a;
    }
    .conf-badge {
        display: inline-block;
        padding: 0.1rem 0.5rem;
        border-radius: 99px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
    }
    div[data-testid="stFileUploader"] {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
    }
    .stSelectbox > div, .stSlider > div {
        background: transparent;
    }
    label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, model_type: str):
    """Load YOLO model from best.pth"""
    try:
        if model_type == "YOLOv5 (ultralytics)":
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=model_path, force_reload=False)
            return model, "yolov5"
        elif model_type == "YOLOv8 (ultralytics)":
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model, "yolov8"
        else:
            # Generic PyTorch — load weights
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict):
                st.warning("Loaded raw state_dict. For full inference, select the correct model type.")
            return model, "generic"
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None


def run_inference_yolov5(model, image: Image.Image, conf_thresh: float, iou_thresh: float):
    model.conf = conf_thresh
    model.iou = iou_thresh
    results = model(image)
    return results


def run_inference_yolov8(model, image: Image.Image, conf_thresh: float, iou_thresh: float):
    results = model(image, conf=conf_thresh, iou=iou_thresh)
    return results


def annotate_yolov5(image: Image.Image, results) -> tuple[np.ndarray, list]:
    img = np.array(image)
    df = results.pandas().xyxy[0]
    detections = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = float(row['confidence'])
        label = row['name']
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 136), 2)
        # Draw label bg
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 136), -1)
        cv2.putText(img, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        detections.append({"label": label, "confidence": conf,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return img, detections


def annotate_yolov8(image: Image.Image, results) -> tuple[np.ndarray, list]:
    img = np.array(image)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 136), 2)
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 136), -1)
            cv2.putText(img, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            detections.append({"label": label, "confidence": conf,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return img, detections


def conf_color(conf: float) -> str:
    if conf >= 0.80: return "#00ff88"
    if conf >= 0.60: return "#ffd700"
    return "#ff6b6b"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model_folder = st.text_input("📁 Model folder path", value="./",
                                  help="Folder containing best.pth")
    model_type = st.selectbox("Model type", [
        "YOLOv8 (ultralytics)",
        "YOLOv5 (ultralytics)",
    ])
    conf_thresh = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

    st.markdown("---")
    load_btn = st.button("🔄 Load Model")

    st.markdown("---")
    st.markdown("""
    <div style='color:#555; font-size:0.75rem; line-height:1.6'>
    <b style='color:#666'>How to use</b><br>
    1. Set folder path to where <code>best.pth</code> lives<br>
    2. Select model type<br>
    3. Click Load Model<br>
    4. Upload an image<br>
    5. Click Detect
    </div>
    """, unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_type_loaded = None

if load_btn:
    model_path = str(Path(model_folder) / "best.pth")
    if not os.path.exists(model_path):
        # also try .pt
        model_path_pt = str(Path(model_folder) / "best.pt")
        if os.path.exists(model_path_pt):
            model_path = model_path_pt
        else:
            st.sidebar.error(f"❌ best.pth not found in:\n`{model_folder}`")
            st.stop()
    with st.sidebar:
        with st.spinner("Loading model..."):
            model, mtype = load_model(model_path, model_type)
    if model is not None:
        st.session_state.model = model
        st.session_state.model_type_loaded = mtype
        st.sidebar.success("✅ Model loaded!")


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e8e8e8; margin-bottom:0; font-size:1.8rem'>
  🔍 Object Detection
</h1>
<p style='color:#555; font-family:DM Sans; margin-top:0.3rem; margin-bottom:2rem'>
  Upload an image → run inference → see detections
</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload image (JPG / PNG / BMP)", type=["jpg","jpeg","png","bmp","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("**Input Image**")
        st.image(image, use_container_width=True)

    detect_btn = st.button("🚀 Detect Objects")

    if detect_btn:
        if st.session_state.model is None:
            st.warning("⚠️ Please load the model first (sidebar → Load Model).")
        else:
            with st.spinner("Running inference..."):
                try:
                    mtype = st.session_state.model_type_loaded
                    model  = st.session_state.model

                    if mtype == "yolov5":
                        results = run_inference_yolov5(model, image, conf_thresh, iou_thresh)
                        annotated, detections = annotate_yolov5(image, results)
                    elif mtype == "yolov8":
                        results = run_inference_yolov8(model, image, conf_thresh, iou_thresh)
                        annotated, detections = annotate_yolov8(image, results)
                    else:
                        st.error("Generic model inference not implemented. Use YOLOv5 or YOLOv8.")
                        st.stop()

                    with col2:
                        st.markdown("**Detected Objects**")
                        st.image(annotated, use_container_width=True)

                        # Download button
                        buf = io.BytesIO()
                        Image.fromarray(annotated).save(buf, format="PNG")
                        st.download_button("⬇️ Download result", buf.getvalue(),
                                           file_name="detection_result.png",
                                           mime="image/png")

                    # ── Metrics ────────────────────────────────────────────
                    from collections import Counter
                    counts = Counter(d["label"] for d in detections)
                    avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0

                    st.markdown("<br>", unsafe_allow_html=True)
                    mcols = st.columns(3)
                    with mcols[0]:
                        st.markdown(f"""
                        <div class='metric-card'>
                          <div class='value'>{len(detections)}</div>
                          <div class='label'>Total Detections</div>
                        </div>""", unsafe_allow_html=True)
                    with mcols[1]:
                        st.markdown(f"""
                        <div class='metric-card'>
                          <div class='value'>{len(counts)}</div>
                          <div class='label'>Unique Classes</div>
                        </div>""", unsafe_allow_html=True)
                    with mcols[2]:
                        st.markdown(f"""
                        <div class='metric-card'>
                          <div class='value'>{avg_conf:.2f}</div>
                          <div class='label'>Avg Confidence</div>
                        </div>""", unsafe_allow_html=True)

                    # ── Class breakdown ────────────────────────────────────
                    if counts:
                        st.markdown("<br>**Class Breakdown**", unsafe_allow_html=True)
                        bcols = st.columns(min(len(counts), 5))
                        for i, (cls, cnt) in enumerate(counts.most_common()):
                            with bcols[i % len(bcols)]:
                                st.markdown(f"""
                                <div class='metric-card'>
                                  <div class='value' style='font-size:1.4rem'>{cnt}</div>
                                  <div class='label'>{cls}</div>
                                </div>""", unsafe_allow_html=True)

                    # ── Detection table ────────────────────────────────────
                    if detections:
                        st.markdown("<br>**All Detections**", unsafe_allow_html=True)
                        rows = ""
                        for i, d in enumerate(sorted(detections, key=lambda x: -x["confidence"])):
                            color = conf_color(d["confidence"])
                            rows += f"""<tr>
                              <td>{i+1}</td>
                              <td>{d['label']}</td>
                              <td><span class='conf-badge' style='background:{color}22; color:{color}'>
                                {d['confidence']:.3f}</span></td>
                              <td style='color:#555'>{d['x1']},{d['y1']} → {d['x2']},{d['y2']}</td>
                            </tr>"""
                        st.markdown(f"""
                        <table class='detection-table'>
                          <thead><tr>
                            <th>#</th><th>Class</th><th>Confidence</th><th>Bounding Box</th>
                          </tr></thead>
                          <tbody>{rows}</tbody>
                        </table>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Inference error: {e}")
                    st.exception(e)

else:
    st.markdown("""
    <div class='upload-box'>
      <div style='font-size:2.5rem'>🖼️</div>
      <div style='color:#555; margin-top:0.5rem'>Upload an image using the uploader above</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem; text-align:center; color:#333; font-size:0.75rem; font-family:Space Mono'>
  YOLO OBJECT DETECTION DASHBOARD
</div>
""", unsafe_allow_html=True)
