import streamlit as st
import cv2
import av
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide"
)

# =========================
# CUSTOM DESIGN
# =========================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #aab4c3;
    font-size: 18px;
}

.info-box {
    background-color: #202b40;
    padding: 18px;
    border-radius: 12px;
    color: #dbeafe;
}

.warning-box {
    background-color: #3a3018;
    padding: 15px;
    border-radius: 10px;
    color: #ffd966;
}

.detected-box {
    background-color: #172b22;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: #b7f7cd;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD YOLO MODEL
# =========================

@st.cache_resource
def load_model():
    return YOLO("yolo26s.pt")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading error: {e}")

# =========================
# OBJECT DETECTION
# =========================

TARGET_OBJECTS = [
    "person",
    "chair",
    "couch",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bench",
    "suitcase",
    "dining table"
]

class VisionProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.last_detection_time = 0
        self.detections = []

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            image,
            imgsz=640,
            conf=0.30,
            verbose=False
        )

        annotated = image.copy()
        detected_objects = []

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                name = self.model.names[class_id]

                if name not in TARGET_OBJECTS:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detected_objects.append(name)

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 100),
                    3
                )

                label = f"{name} {confidence:.0%}"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 100),
                    2
                )

        self.detections = list(set(detected_objects))

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">👁️ VisionGuide AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered visual assistance prototype</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# PRIVACY WARNING
# =========================

st.markdown("""
<div class="warning-box">
⚠️ <b>Privacy Notice:</b> Camera images are sent to the VisionGuide server
for AI processing. Start the camera only if you agree.
<br><br>
This is an experimental prototype. Do not rely on it for navigation or safety.
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# CAMERA SECTION
# =========================

st.subheader("📷 Live Camera")

if not model_loaded:

    st.error("YOLO model could not be loaded.")

else:

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    camera = webrtc_streamer(
        key="visionguide-camera",
        video_processor_factory=VisionProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=False
    )

# =========================
# DETECTION INFORMATION
# =========================

st.divider()

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("🔍 Detection Information")

    if model_loaded and camera.video_processor:

        detections = camera.video_processor.detections

        if detections:

            for item in detections:

                st.markdown(
                    f"""
                    <div class="detected-box">
                    🟢 <b>{item.capitalize()}</b> detected
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No target objects detected currently."
            )

    else:

        st.info(
            "Start the camera to begin object detection."
        )

with col2:

    st.subheader("⚙️ System Status")

    if model_loaded:

        st.success("YOLO Model Loaded")

    else:

        st.error("Model Not Loaded")

    if camera.state.playing:

        st.success("Camera Running")

    else:

        st.info("Camera Stopped")

    st.caption(
        "Detection runs on the server."
    )

# =========================
# FOOTER
# =========================

st.divider()

st.caption("""
VisionGuide AI | Experimental AI visual assistance prototype

Not a replacement for a cane, guide dog, trained mobility assistance,
or human assistance. Detection can be incorrect or delayed.
""")
