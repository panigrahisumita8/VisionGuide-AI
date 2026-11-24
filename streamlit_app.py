
import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import json
import threading
from collections import defaultdict, deque

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "weights.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.45
MAX_DETECTIONS = 20
ANNOUNCEMENT_COOLDOWN = 6
STABILITY_FRAMES = 3

# --------------------------------------------------
# CUSTOM GUI
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111827;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label, span {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid #29354c;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111a2b;
    border: 1px solid #29354c;
    border-radius: 12px;
}
.stButton button {
    border-radius: 9px;
}
hr {
    border-color: #29354c;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    class_names = model.names

    if isinstance(class_names, list):
        class_names = dict(enumerate(class_names))

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info("Check that weights.pt is present in the repository root.")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("# 👁️ VisionGuide AI")
st.markdown("### Real-time visual assistance prototype")
st.caption("AI-powered object recognition and voice-based scene descriptions")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 👁️ VisionGuide")
    st.caption("AI Vision Assistant")

    st.divider()

    st.markdown("### ⚙️ Detection settings")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.20,
        max_value=0.90,
        value=CONFIDENCE,
        step=0.05
    )

    voice_enabled = st.toggle(
        "Voice announcements",
        value=True
    )

    st.divider()

    st.markdown("### 🧠 Model information")

    st.success("Custom YOLO model loaded")

    st.metric("Trained classes", len(class_names))

    st.caption("Checkpoint: weights.pt")
    st.caption("Input size: 640 × 640")

    st.divider()

    st.caption("VisionGuide AI | Experimental prototype")

# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class VisionGuideProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.lock = threading.Lock()

        self.frame_count = 0
        self.history = defaultdict(
            lambda: deque(maxlen=5)
        )

        self.last_spoken = {}

        self.latest_detections = []
        self.latest_message = ""
        self.message_id = 0
        self.last_error = ""

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every second frame to reduce CPU load
        if self.frame_count % 2 != 0:
            return frame

        try:
            results = self.model.predict(
                image,
                imgsz=IMAGE_SIZE,
                conf=confidence,
                max_det=MAX_DETECTIONS,
                verbose=False
            )

            height, width = image.shape[:2]

            detections = []
            present_objects = set()

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    # Use the checkpoint's original class mapping
                    label = str(self.model.names[class_id])

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    # Determine approximate screen position
                    center_x = (x1 + x2) / 2

                    if center_x < width * 0.35:
                        position = "left"
                    elif center_x > width * 0.65:
                        position = "right"
                    else:
                        position = "ahead"

                    detections.append({
                        "name": label,
                        "class_id": class_id,
                        "confidence": score,
                        "position": position
                    })

                    present_objects.add((label, position))

                    # Draw detection box
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 130),
                        2
                    )

                    text = f"{label} {score:.0%}"

                    cv2.putText(
                        image,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 220, 130),
                        2
                    )

            # Update stability history
            all_keys = set(self.history.keys()) | present_objects

            for key in all_keys:
                self.history[key].append(
                    1 if key in present_objects else 0
                )

            # Prepare voice announcements
            current_time = time.time()
            new_message = None

            for label, position in present_objects:

                key = (label, position)

                if sum(self.history[key]) < STABILITY_FRAMES:
                    continue

                last_time = self.last_spoken.get(key, 0)

                if current_time - last_time < ANNOUNCEMENT_COOLDOWN:
                    continue

                new_message = f"{label} on your {position}"
                self.last_spoken[key] = current_time
                break

            with self.lock:

                self.latest_detections = detections
                self.last_error = ""

                if new_message:
                    self.latest_message = new_message
                    self.message_id += 1

        except Exception as error:

            with self.lock:
                self.last_error = str(error)

        return frame.from_ndarray(
            image,
            format="bgr24"
        )

    def get_data(self):

        with self.lock:
            return (
                list(self.latest_detections),
                self.latest_message,
                self.message_id,
                self.last_error
            )

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

left, right = st.columns([2, 1])

with left:
    st.markdown("## 📷 Live camera")
    st.caption("Allow camera access when your browser asks.")

with right:
    st.markdown("## 🎙️ Voice assistant")
    st.caption(
        "Voice announcements are approximate screen positions, "
        "not distance measurements."
    )

# --------------------------------------------------
# WEBRTC CAMERA
# --------------------------------------------------

ctx = webrtc_streamer(
    key="visionguide-live-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VisionGuideProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

st.divider()

# --------------------------------------------------
# LIVE DASHBOARD
# --------------------------------------------------

@st.fragment(run_every=0.7)
def live_dashboard():

    if not ctx.state.playing or ctx.video_processor is None:

        st.info("📷 Start the camera above to begin object detection.")

        col1, col2, col3 = st.columns(3)

        col1.metric("Camera", "Inactive")
        col2.metric("Detection", "Waiting")
        col3.metric("Voice", "Waiting")

        return

    detections, message, message_id, error = (
        ctx.video_processor.get_data()
    )

    if error:
        st.error(f"Detection error: {error}")

    # Status metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Camera", "Active")
    col2.metric("Objects detected", len(detections))
    col3.metric(
        "Voice assistant",
        "Enabled" if voice_enabled else "Disabled"
    )

    st.divider()

    # Latest announcement
    st.markdown("### 🔊 Latest announcement")

    if message:
        st.info(f"🗣️ {message}")
    else:
        st.write("Waiting for a stable object detection...")

    # Speak only new announcements
    if (
        voice_enabled
        and message
        and message_id != st.session_state.get("last_message_id", -1)
    ):

        st.session_state.last_message_id = message_id

        components.html(
            f"""
            <script>
            const message = {json.dumps(message)};

            if ("speechSynthesis" in window) {{
                window.speechSynthesis.cancel();

                const speech = new SpeechSynthesisUtterance(message);
                speech.rate = 0.95;
                speech.pitch = 1;
                speech.volume = 1;

                window.speechSynthesis.speak(speech);
            }}
            </script>
            """,
            height=0
        )

    st.divider()

    # Detection list
    st.markdown("### 🎯 Detected objects")

    if detections:

        for item in detections:

            with st.container(border=True):

                col1, col2, col3 = st.columns([2, 1, 1])

                col1.markdown(
                    f"**{item['name']}**"
                )

                col2.caption(
                    f"ID: {item['class_id']}"
                )

                col3.caption(
                    f"{item['confidence']:.0%}"
                )

                st.caption(
                    f"Approximate position: {item['position'].title()}"
                )

    else:
        st.info("No objects detected in the latest processed frame.")

live_dashboard()

# --------------------------------------------------
# MODEL CLASS INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("🧠 View all trained model classes"):

    st.caption(
        "Class IDs and names are read directly from weights.pt."
    )

    class_columns = st.columns(3)

    for index, (class_id, class_name) in enumerate(
        class_names.items()
    ):

        with class_columns[index % 3]:
            st.write(f"**{class_id}** · {class_name}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "VisionGuide AI is an experimental educational prototype. "
    "It can make incorrect detections and cannot measure distance "
    "or guarantee obstacle avoidance. Do not rely on it as a "
    "substitute for a mobility aid or human assistance."
)
