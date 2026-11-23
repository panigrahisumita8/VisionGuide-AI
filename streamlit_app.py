
import streamlit as st
import cv2
import av
import time
import threading
import numpy as np

from collections import deque
from huggingface_hub import hf_hub_download
from rfdetr import RFDETRMedium
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide"
)

# ==================================================
# CONFIGURATION
# ==================================================

HF_REPO_ID = "ommppra/visionguide-rfdetr"
HF_FILENAME = "weights.pt"

IMAGE_SIZE = 576
CONFIDENCE = 0.40
VOICE_COOLDOWN = 6

# Class IDs from your Roboflow data.yaml.
# Verify these against the trained model before relying
# on the displayed object names.

CLASS_NAMES = [
    "Backpack",
    "Bed",
    "Bench",
    "Bicycle",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Door",
    "Handbag",
    "Laptop",
    "Mobile Phone",
    "Motorcycle",
    "Person",
    "Pothole",
    "Potted Plant",
    "Refrigerator",
    "Sink",
    "Sofa",
    "Stairs",
    "Stop Sign",
    "Table",
    "Traffic Light",
    "Truck",
    "Umbrella",
    "Wall",
    "Window",
]

# ==================================================
# LOAD RF-DETR MODEL
# ==================================================

@st.cache_resource(show_spinner="Downloading and loading VisionGuide AI model...")
def load_model():

    token = st.secrets["HF_TOKEN"]

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=token,
    )

    model = RFDETRMedium(
        pretrain_weights=checkpoint_path,
        resolution=IMAGE_SIZE,
    )

    return model


try:
    model = load_model()
    MODEL_READY = True
except Exception as error:
    MODEL_READY = False
    st.error("Could not load the RF-DETR model.")
    st.exception(error)

# ==================================================
# VISIONGUIDE PROCESSOR
# ==================================================

class VisionGuideProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()
        self.detections = []
        self.speech_queue = deque(maxlen=10)
        self.last_announced = {}
        self.frame_count = 0

    # ==================================================
    # VOICE ANNOUNCEMENTS
    # ==================================================

    def announce(self, detections):

        now = time.time()
        announcements = []
        announced_keys = set()

        for item in detections:

            name = item["name"]
            position = item["position"]
            confidence = item["confidence"]

            key = (name, position)

            previous_time = self.last_announced.get(key, 0)

            if now - previous_time >= VOICE_COOLDOWN:

                if position == "centre":
                    direction = "ahead"
                else:
                    direction = "to your " + position

                message = f"{name} {direction}."

                if confidence < 0.60:
                    message = (
                        f"Possible {name} "
                        f"{direction}."
                    )

                announcements.append(message)
                announced_keys.add(key)

            if len(announcements) >= 3:
                break

        if announcements:

            with self.lock:
                self.speech_queue.append(
                    " ".join(announcements)
                )

            for key in announced_keys:
                self.last_announced[key] = now

    # ==================================================
    # CAMERA FRAME PROCESSING
    # ==================================================

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every second frame
        if self.frame_count % 2 == 0 and MODEL_READY:

            try:
                rgb_image = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                results = model.predict(
                    rgb_image,
                    threshold=CONFIDENCE,
                )

                current_detections = []

                for index in range(len(results.xyxy)):

                    x1, y1, x2, y2 = map(
                        int,
                        results.xyxy[index]
                    )

                    confidence = float(
                        results.confidence[index]
                    )

                    class_id = int(
                        results.class_id[index]
                    )

                    if not 0 <= class_id < len(CLASS_NAMES):
                        continue

                    name = CLASS_NAMES[class_id]

                    center_x = (x1 + x2) / 2
                    width = img.shape[1]

                    if center_x < width / 3:
                        position = "left"
                    elif center_x > width * 2 / 3:
                        position = "right"
                    else:
                        position = "centre"

                    current_detections.append({
                        "name": name,
                        "position": position,
                        "confidence": confidence,
                    })

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = (
                        f"{name} {confidence:.2f}"
                        f" - {position}"
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                with self.lock:
                    self.detections = current_detections

                if current_detections:
                    self.announce(current_detections)

            except Exception as error:
                with self.lock:
                    self.detections = [{
                        "name": "Detection error",
                        "position": str(error)[:80],
                        "confidence": 0.0,
                    }]

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )

    # ==================================================
    # GET DETECTIONS
    # ==================================================

    def get_detections(self):

        with self.lock:
            return list(self.detections)

    # ==================================================
    # GET SPEECH
    # ==================================================

    def get_speech(self):

        with self.lock:
            if self.speech_queue:
                return self.speech_queue.popleft()

            return None


# ==================================================
# INTERFACE DESIGN
# ==================================================

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 750;
    margin-bottom: 0;
}
.subtitle {
    color: #888888;
    font-size: 15px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">👁️ VisionGuide AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered object detection assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

if not MODEL_READY:
    st.warning(
        "The model could not be loaded. "
        "Check your Hugging Face token and model repository."
    )
    st.stop()

# ==================================================
# DASHBOARD
# ==================================================

left, right = st.columns([2, 1], gap="large")

# ==================================================
# CAMERA
# ==================================================

with left:

    st.subheader("📷 Live Camera")

    ctx = webrtc_streamer(
        key="visionguide-camera",
        video_processor_factory=VisionGuideProcessor,

        rtc_configuration=RTCConfiguration({
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }),

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        async_processing=True,
    )

    st.caption(
        "Click Start to activate the camera "
        "and begin object detection."
    )

# ==================================================
# DASHBOARD
# ==================================================

@st.fragment(run_every=0.5)
def dashboard():

    with right:

        st.subheader("Controls")

        if ctx.state.playing:
            st.success("Detection is running")
        else:
            st.warning("Detection is stopped")

        st.divider()

        st.subheader("🔊 Voice Assistant")
        st.write("Voice announcements: Enabled")

        st.divider()

        st.subheader("Detected Objects")

        if ctx.video_processor:

            detections = (
                ctx.video_processor.get_detections()
            )

            if detections:

                for item in detections:

                    st.write(
                        f"**{item['name'].title()}**"
                        f" · {item['position'].title()}"
                        f" · {item['confidence']:.0%}"
                    )

            else:
                st.caption("No objects detected yet.")

            message = ctx.video_processor.get_speech()

            if message:

                escaped_message = (
                    message
                    .replace("\\", "\\\\")
                    .replace("'", "\\'")
                    .replace("\n", " ")
                )

                st.components.v1.html(
                    f"""
                    <script>
                    const speech =
                        new SpeechSynthesisUtterance(
                            '{escaped_message}'
                        );

                    speech.rate = 1;
                    speech.volume = 1;

                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(speech);
                    </script>
                    """,
                    height=0,
                )

        else:
            st.caption(
                "Start the camera to detect objects."
            )

        st.divider()

        st.caption("Supported classes: 30")
        st.caption(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")


dashboard()

st.divider()

st.warning(
    "VisionGuide AI is an experimental demonstration. "
    "It can miss or misidentify objects. Do not rely on it "
    "for mobility, road crossing, or avoiding hazards."
)
