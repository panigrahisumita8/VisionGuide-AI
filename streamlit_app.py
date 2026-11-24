
import streamlit as st
import cv2
import time
import json
import threading
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)
import streamlit.components.v1 as components


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide"
)

MODEL_PATH = "weights.pt"

IMAGE_SIZE = 640
CONFIDENCE = 0.50
ANNOUNCEMENT_COOLDOWN = 8
STABILITY_FRAMES = 3


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()


# IMPORTANT:
# Use the checkpoint's own class mapping.
# Never manually assign class IDs here.

CLASS_NAMES = model.names

if isinstance(CLASS_NAMES, list):
    CLASS_NAMES = dict(enumerate(CLASS_NAMES))

st.title("👁️ VisionGuide AI")
st.caption("Real-time object detection and voice assistance")


# ---------------- VIDEO PROCESSOR ----------------

class VisionGuideProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = model
        self.lock = threading.Lock()

        self.history = defaultdict(
            lambda: deque(maxlen=5)
        )

        self.last_spoken = {}
        self.latest_detections = []
        self.latest_message = ""
        self.message_id = 0

        self.frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Run detection every second frame
        if self.frame_count % 2 == 0:

            results = self.model.predict(
                img,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=0.50,
                max_det=20,
                verbose=False
            )

            detections = []
            present_objects = set()

            height, width = img.shape[:2]

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    # Get the actual trained class name
                    label = self.model.names[class_id]

                    # Ignore uncertain detections
                    if confidence < CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    center_x = (x1 + x2) / 2

                    if center_x < width * 0.35:
                        position = "left"
                    elif center_x > width * 0.65:
                        position = "right"
                    else:
                        position = "ahead"

                    detections.append({
                        "name": str(label),
                        "class_id": class_id,
                        "confidence": confidence,
                        "position": position
                    })

                    present_objects.add((str(label), position))

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    text = f"{label} {confidence:.2f}"

                    cv2.putText(
                        img,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # Update detection stability history
            all_keys = set(self.history.keys()) | present_objects

            for key in all_keys:
                self.history[key].append(
                    1 if key in present_objects else 0
                )

            # Find stable detections
            current_time = time.time()

            for label, position in present_objects:

                key = (label, position)

                stable_count = sum(self.history[key])

                if stable_count < STABILITY_FRAMES:
                    continue

                previous_time = self.last_spoken.get(key, 0)

                if current_time - previous_time < ANNOUNCEMENT_COOLDOWN:
                    continue

                message = f"{label} {position}"

                with self.lock:
                    self.latest_message = message
                    self.message_id += 1

                self.last_spoken[key] = current_time

            with self.lock:
                self.latest_detections = detections

        return frame.from_ndarray(img, format="bgr24")

    def get_data(self):

        with self.lock:
            return (
                list(self.latest_detections),
                self.latest_message,
                self.message_id
            )


# ---------------- CAMERA ----------------

st.subheader("Live camera")

ctx = webrtc_streamer(
    key="visionguide-camera",
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


# ---------------- DETECTION DASHBOARD ----------------

@st.fragment(run_every=0.5)
def dashboard():

    if not ctx.state.playing or ctx.video_processor is None:
        st.info("Start the camera to begin detection.")
        return

    detections, message, message_id = (
        ctx.video_processor.get_data()
    )

    st.subheader("Detected objects")

    if detections:

        for item in detections:

            st.write(
                f"**{item['name']}** | "
                f"Class ID: {item['class_id']} | "
                f"Confidence: {item['confidence']:.0%} | "
                f"Position: {item['position']}"
            )

    else:
        st.write("No objects detected in the latest processed frame.")

    # Speak only new announcements
    if message and message_id != st.session_state.get(
        "last_message_id", -1
    ):

        st.session_state.last_message_id = message_id

        components.html(
            f"""
            <script>
            const message = {json.dumps(message)};
            if ("speechSynthesis" in window) {{
                window.speechSynthesis.cancel();
                const speech = new SpeechSynthesisUtterance(message);
                speech.rate = 1;
                speech.pitch = 1;
                speech.volume = 1;
                window.speechSynthesis.speak(speech);
            }}
            </script>
            """,
            height=0
        )


dashboard()


# ---------------- MODEL CLASS INFORMATION ----------------

with st.expander("View trained model class IDs"):

    st.write(
        "These IDs and names are read directly from your checkpoint."
    )

    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"**{class_id}** : {class_name}")


st.divider()

st.caption(
    "VisionGuide AI is an experimental prototype. "
    "Do not rely on it as a substitute for a mobility aid or human assistance."
)
