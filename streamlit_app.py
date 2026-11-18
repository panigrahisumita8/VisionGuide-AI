import json
import os
import threading
import time
import uuid

import av
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .vg-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
    .vg-subtitle {color: #8b949e; margin-top: 0.2rem; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="vg-title">👁️ VisionGuide AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vg-subtitle">AI-powered object detection assistant</div>',
    unsafe_allow_html=True,
)
st.divider()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv("VISIONGUIDE_MODEL", "yolo26s.pt")
IMAGE_SIZE = 640
CONFIDENCE = 0.25
VOICE_COOLDOWN = 6

# Keep this list focused on common, larger objects.
TARGET_OBJECTS = {
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
    "dining table",
}

# Browser speech is generated from these short, camera-relative messages.
# It does not estimate distance, motion, or whether a route is safe.


# ============================================================
# LOAD MODEL ONCE PER SERVER
# ============================================================

@st.cache_resource(show_spinner="Loading VisionGuide AI model...")
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found. "
            "Add the model file to the deployed app or configure VISIONGUIDE_MODEL."
        )
    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the AI model: {exc}")
    st.info(
        "Check that the model file is included in your deployment and that "
        "requirements.txt contains ultralytics."
    )
    st.stop()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VisionGuideProcessor:
    """Runs YOLO on frames received from the visitor's browser camera."""

    def __init__(self, yolo_model):
        self.model = yolo_model
        self.lock = threading.Lock()
        self.detections = []
        self.last_announcement = ""
        self.announcement_id = ""
        self.last_announced_at = {}
        self.frame_number = 0
        self.error = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_number += 1

        # Process every second frame to reduce server load.
        if self.frame_number % 2 == 0:
            try:
                results = self.model.predict(
                    image,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    verbose=False,
                    device="cpu",
                )

                current_detections = []
                announcement_parts = []
                now = time.time()

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        name = self.model.names[class_id]

                        if name not in TARGET_OBJECTS:
                            continue

                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        # Ignore very small boxes to reduce noisy alerts.
                        box_area = max(0, x2 - x1) * max(0, y2 - y1)
                        frame_area = image.shape[0] * image.shape[1]
                        if box_area < frame_area * 0.015:
                            continue

                        center_x = (x1 + x2) / 2

                        if center_x < image.shape[1] / 3:
                            position = "left"
                        elif center_x > image.shape[1] * 2 / 3:
                            position = "right"
                        else:
                            position = "centre"

                        current_detections.append(
                            {
                                "name": name,
                                "position": position,
                                "confidence": round(confidence, 2),
                            }
                        )

                        cv2.rectangle(
                            image, (x1, y1), (x2, y2), (0, 220, 90), 2
                        )
                        label = f"{name} {confidence:.2f} - {position}"
                        cv2.putText(
                            image,
                            label,
                            (x1, max(24, y1 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 220, 90),
                            2,
                        )

                        key = (name, position)
                        previous = self.last_announced_at.get(key, 0)
                        if now - previous >= VOICE_COOLDOWN:
                            announcement_parts.append(
                                f"{name} detected on the {position}"
                            )
                            self.last_announced_at[key] = now

                # Avoid repeating the same object-position multiple times
                # in one spoken announcement.
                announcement_parts = list(dict.fromkeys(announcement_parts))[:3]

                with self.lock:
                    self.detections = current_detections
                    self.error = ""

                    if announcement_parts:
                        self.last_announcement = ". ".join(announcement_parts) + "."
                        self.announcement_id = str(uuid.uuid4())

            except Exception as exc:
                with self.lock:
                    self.error = str(exc)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def snapshot(self):
        with self.lock:
            return (
                list(self.detections),
                self.last_announcement,
                self.announcement_id,
                self.error,
            )


# ============================================================
# BROWSER SPEECH
# ============================================================

def speak_in_browser(message: str, event_id: str):
    """Speak a new announcement in the visitor's browser, not on the server."""
    safe_message = json.dumps(message)
    safe_event_id = json.dumps(event_id)

    html = f"""
    <script>
    const message = {safe_message};
    const eventId = {safe_event_id};

    if (message && eventId && "speechSynthesis" in window) {{
        const storageKey = "visionguide_last_spoken_event";
        let previousEvent = "";
        try {{
            previousEvent = window.localStorage.getItem(storageKey) || "";
        }} catch (e) {{}}

        if (eventId !== previousEvent) {{
            try {{
                window.localStorage.setItem(storageKey, eventId);
            }} catch (e) {{}}

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = "en-US";
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }}
    }}
    </script>
    """
    components.html(html, height=0, width=0)


# ============================================================
# LIVE CAMERA
# ============================================================

st.subheader("📷 Live Camera")

st.warning(
    "Camera images are sent from your browser to the VisionGuide server "
    "for AI processing. Start the camera only if you agree to this."
)

ctx = webrtc_streamer(
    key="visionguide-live-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VisionGuideProcessor(model),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    desired_playing_state=True,
)

st.caption(
    "Allow camera permission in your browser. Use the Start button in the "
    "camera panel to begin and Stop to pause."
)


# ============================================================
# DASHBOARD / LIVE RESULTS
# ============================================================

@st.fragment(run_every=0.5)
def live_dashboard():
    left, right = st.columns([2, 1], gap="large")

    processor = ctx.video_processor if ctx.state.playing else None

    with left:
        st.subheader("Detection Preview")
        if ctx.state.playing:
            st.success("Camera connected")
            st.caption("The live annotated camera feed appears in the camera panel above.")
        else:
            st.info("Camera is stopped. Start the camera in the panel above.")

    with right:
        st.subheader("System Status")
        if ctx.state.playing and processor is not None:
            st.success("VisionGuide is processing camera frames")
        elif ctx.state.signalling:
            st.info("Connecting to camera...")
        else:
            st.warning("Camera is not running")

        st.divider()
        st.subheader("🔊 Voice Assistant")
        st.write("Browser voice announcements: Enabled")
        st.caption("Uses the visitor's browser speech engine.")

        st.divider()
        st.subheader("Detected Objects")

        if processor is None:
            st.caption("No active camera stream.")
            return

        detections, announcement, announcement_id, processor_error = processor.snapshot()

        if processor_error:
            st.error(f"Detection error: {processor_error}")

        if detections:
            for item in detections:
                st.write(
                    f"**{item['name'].title()}**"
                    f" · {item['position'].title()}"
                    f" · {item['confidence']:.0%}"
                )
        else:
            st.caption("No selected objects detected in the latest processed frame.")

        if announcement and announcement_id:
            speak_in_browser(announcement, announcement_id)

        st.divider()
        st.caption(f"Selected object classes: {len(TARGET_OBJECTS)}")
        st.caption(f"Detection image size: {IMAGE_SIZE}px")


live_dashboard()

st.divider()
st.caption(
    "VisionGuide AI is an experimental object-detection prototype. "
    "It cannot measure distance, reliably detect every hazard, or determine "
    "whether a path is safe. Do not rely on it for navigation."
)
