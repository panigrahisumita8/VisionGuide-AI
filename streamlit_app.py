
import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import json
import threading
import queue
import av

from collections import defaultdict, deque
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "weights.pt"
IMAGE_SIZE = 416
CONFIDENCE = 0.45
MAX_DETECTIONS = 15
ANNOUNCEMENT_COOLDOWN = 6
STABILITY_FRAMES = 3

# ==================================================
# 2. GUI DESIGN
# ==================================================

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
.stButton button {
    border-radius: 9px;
}
hr {
    border-color: #29354c;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. LOAD CUSTOM MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

try:
    model = load_model()
    class_names = model.names

    if isinstance(class_names, list):
        class_names = dict(enumerate(class_names))

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ==================================================
# 4. HEADER
# ==================================================

st.markdown("# 👁️ VisionGuide AI")
st.markdown("### Real-time visual assistance prototype")
st.caption("AI-powered object recognition and voice assistance")

st.divider()

# ==================================================
# 5. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 👁️ VisionGuide")
    st.caption("AI Vision Assistant")

    st.divider()

    st.markdown("### Detection settings")

    confidence = st.slider(
        "Detection confidence",
        0.20,
        0.90,
        CONFIDENCE,
        0.05
    )

    voice_enabled = st.toggle(
        "Voice announcements",
        value=True
    )

    st.divider()

    st.markdown("### Model information")

    st.success("Custom YOLO model loaded")

    st.metric("Trained classes", len(class_names))

    st.caption("Checkpoint: weights.pt")
    st.caption("Image size: 416")

    st.divider()

    st.caption("VisionGuide AI | Experimental prototype")

# ==================================================
# 6. BACKGROUND VIDEO PROCESSOR
# ==================================================

class VisionGuideProcessor(VideoProcessorBase):

    def __init__(self, confidence_value):

        self.model = model
        self.confidence = confidence_value

        self.lock = threading.Lock()

        # Queue holds only the newest frame
        self.frame_queue = queue.Queue(maxsize=1)

        self.latest_image = None
        self.latest_detections = []
        self.latest_message = ""

        self.message_id = 0
        self.last_error = ""

        self.frame_count = 0

        self.history = defaultdict(
            lambda: deque(maxlen=5)
        )

        self.last_spoken = {}

        # Start background AI worker
        self.worker = threading.Thread(
            target=self.detection_worker,
            daemon=True
        )

        self.worker.start()

    # ------------------------------------------------
    # CAMERA THREAD: NEVER WAIT FOR YOLO
    # ------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        # Send newest frame to background worker
        try:
            self.frame_queue.put_nowait(image.copy())

        except queue.Full:

            # Remove old frame
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.frame_queue.put_nowait(image.copy())
            except queue.Full:
                pass

        # Read latest processed image
        with self.lock:

            if self.latest_image is not None:
                output = self.latest_image.copy()
            else:
                output = image

        # Return immediately, without waiting for YOLO
        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

    # ------------------------------------------------
    # BACKGROUND DETECTION WORKER
    # ------------------------------------------------

    def detection_worker(self):

        while True:

            try:
                image = self.frame_queue.get(timeout=2)

            except queue.Empty:
                continue

            try:
                annotated, detections, message = (
                    self.process_image(image)
                )

                with self.lock:

                    self.latest_image = annotated
                    self.latest_detections = detections
                    self.last_error = ""

                    if message:

                        self.latest_message = message
                        self.message_id += 1

            except Exception as error:

                with self.lock:
                    self.last_error = str(error)

    # ------------------------------------------------
    # YOLO INFERENCE
    # ------------------------------------------------

    def process_image(self, image):

        height, width = image.shape[:2]

        results = self.model.predict(
            image,
            imgsz=IMAGE_SIZE,
            conf=self.confidence,
            max_det=MAX_DETECTIONS,
            verbose=False
        )

        detections = []
        present_objects = set()

        annotated = image.copy()

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                # Use actual checkpoint class names
                label = str(self.model.names[class_id])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Approximate screen position
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

                # Draw bounding boxes
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 130),
                    2
                )

                cv2.putText(
                    annotated,
                    f"{label} {score:.0%}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 130),
                    2
                )

        # Update detection stability
        all_keys = set(self.history.keys()) | present_objects

        for key in all_keys:

            self.history[key].append(
                1 if key in present_objects else 0
            )

        # Generate voice announcement
        message = None
        current_time = time.time()

        for label, position in present_objects:

            key = (label, position)

            if sum(self.history[key]) < STABILITY_FRAMES:
                continue

            last_time = self.last_spoken.get(key, 0)

            if (
                current_time - last_time
                < ANNOUNCEMENT_COOLDOWN
            ):
                continue

            message = f"{label} on your {position}"

            self.last_spoken[key] = current_time

            break

        return annotated, detections, message

    # ------------------------------------------------
    # SHARE DATA WITH STREAMLIT
    # ------------------------------------------------

    def get_data(self):

        with self.lock:

            return (
                list(self.latest_detections),
                self.latest_message,
                self.message_id,
                self.last_error
            )


# ==================================================
# 7. CAMERA INTERFACE
# ==================================================

left, right = st.columns([2, 1])

with left:

    st.markdown("## 📷 Live camera")
    st.caption("Start the camera and allow browser permission.")

with right:

    st.markdown("## 🎙️ Voice assistant")
    st.caption("Real-time object announcements")

ctx = webrtc_streamer(
    key="visionguide-live-camera",
    mode=WebRtcMode.SENDRECV,

    video_processor_factory=lambda: VisionGuideProcessor(
        confidence
    ),

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

st.divider()

# ==================================================
# 8. LIVE DASHBOARD
# ==================================================

@st.fragment(run_every=1.0)
def live_dashboard():

    if (
        not ctx.state.playing
        or ctx.video_processor is None
    ):

        st.info(
            "📷 Start your camera above to begin detection."
        )

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

    col1, col2, col3 = st.columns(3)

    col1.metric("Camera", "Active")

    col2.metric(
        "Objects detected",
        len(detections)
    )

    col3.metric(
        "Voice assistant",
        "Enabled" if voice_enabled else "Disabled"
    )

    st.divider()

    # Latest voice announcement
    st.markdown("### 🔊 Latest announcement")

    if message:
        st.info(f"🗣️ {message}")
    else:
        st.write("Waiting for object detection...")

    # Browser speech
    if (
        voice_enabled
        and message
        and message_id != st.session_state.get(
            "last_message_id", -1
        )
    ):

        st.session_state.last_message_id = message_id

        components.html(
            f"""
            <script>
            const message = {json.dumps(message)};

            if ("speechSynthesis" in window) {{

                window.speechSynthesis.cancel();

                const speech =
                    new SpeechSynthesisUtterance(message);

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

    # Object detection list
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
                    f"Position: {item['position'].title()}"
                )

    else:

        st.info(
            "No objects detected in the latest processed frame."
        )


live_dashboard()

# ==================================================
# 9. MODEL CLASSES
# ==================================================

st.divider()

with st.expander("🧠 View all 68 trained classes"):

    columns = st.columns(3)

    for index, (class_id, class_name) in enumerate(
        class_names.items()
    ):

        with columns[index % 3]:

            st.write(
                f"**{class_id}** · {class_name}"
            )

# ==================================================
# 10. FOOTER
# ==================================================

st.divider()

st.caption(
    "VisionGuide AI is an experimental educational prototype. "
    "It may make incorrect detections and cannot measure distance "
    "or guarantee obstacle avoidance. Do not use it as a substitute "
    "for a mobility aid or human assistance."
)
