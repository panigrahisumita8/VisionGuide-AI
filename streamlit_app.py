import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque

import av
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# FIREBASE ANALYTICS
# ============================================================

FIREBASE_MEASUREMENT_ID = "G-XSHL6SPZ9G"

st.html(
    f"""
    <script>
    (function() {{
        const measurementId = "{FIREBASE_MEASUREMENT_ID}";

        if (window.parent.__visionGuideAnalyticsLoaded) return;
        window.parent.__visionGuideAnalyticsLoaded = true;

        const script = document.createElement("script");
        script.async = true;
        script.src =
            "https://www.googletagmanager.com/gtag/js?id=" + measurementId;

        window.parent.document.head.appendChild(script);

        window.parent.dataLayer = window.parent.dataLayer || [];

        function gtag() {{
            window.parent.dataLayer.push(arguments);
        }}

        window.parent.gtag = gtag;

        gtag("js", new Date());
        gtag("config", measurementId);
    }})();
    </script>
    """,
    unsafe_allow_javascript=True
)


# ============================================================
# THEME AND CSS
# ============================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b1020 !important;
        color: #e5e7eb !important;
    }

    [data-testid="stHeader"] {
        background-color: #0b1020 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #111827 !important;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    label {
        color: #e5e7eb !important;
    }

    h1, h2, h3, h4 {
        color: #f9fafb !important;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263247;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    .vg-card {
        background: #111827;
        border: 1px solid #263247;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
    }

    .vg-title {
        font-size: 32px;
        font-weight: 800;
        color: #60a5fa;
    }

    .vg-subtitle {
        color: #9ca3af;
        font-size: 15px;
    }

    .vg-object {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px;
        margin: 6px 0;
        color: #f8fafc;
    }

    .stButton button {
        border-radius: 10px;
    }

    [data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #263247;
        border-radius: 10px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.getenv("VISIONGUIDE_MODEL", "weights.pt")

IMAGE_SIZE = 416
DEFAULT_CONFIDENCE = 0.25
VOICE_COOLDOWN = 5
STABILITY_REQUIRED = 2
MAX_DETECTIONS = 50

SILENT_CLASSES = {
    "bottle",
    "cup",
    "sink",
    "mouse",
    "pen",
    "tubelight"
}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="vg-card">
        <div class="vg-title">👁️ VisionGuide AI</div>
        <div class="vg-subtitle">
            AI-powered real-time object recognition and voice assistance
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.15,
    max_value=0.90,
    value=DEFAULT_CONFIDENCE,
    step=0.05
)

voice_enabled = st.sidebar.toggle(
    "Enable voice announcements",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.caption("VisionGuide AI")
st.sidebar.caption("Experimental object detection prototype")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


try:
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure weights.pt is uploaded to your repository."
        )
        st.stop()

    model = load_model(MODEL_PATH)

except Exception as error:
    st.error(f"Unable to load the YOLO model: {error}")
    st.stop()


class_names = model.names

if isinstance(class_names, list):
    class_names = {
        index: name
        for index, name in enumerate(class_names)
    }


# ============================================================
# VISIONGUIDE PROCESSOR
# ============================================================

class VisionGuideProcessor:

    def __init__(self):
        self.model = model

        self.confidence = confidence

        self.lock = threading.Lock()

        self.latest_frame = None

        self.latest_detections = []

        self.last_announcement = "Waiting for objects..."

        self.pending_speech = deque(maxlen=60)

        self.announcement_times = {}

        self.stability = defaultdict(
            lambda: deque(maxlen=4)
        )

        self.error = None

        self.last_time = 0

        self.running = True

        self.total_detections = 0

        self.worker = threading.Thread(
            target=self.detection_worker,
            daemon=True
        )

        self.worker.start()

    # --------------------------------------------------------
    # RECEIVE CAMERA FRAME
    # --------------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = image.copy()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    # --------------------------------------------------------
    # DETECTION WORKER
    # --------------------------------------------------------

    def detection_worker(self):

        while self.running:

            try:

                with self.lock:

                    if self.latest_frame is None:
                        frame = None
                    else:
                        frame = self.latest_frame.copy()
                        self.latest_frame = None

                if frame is None:
                    time.sleep(0.03)
                    continue

                results = self.model.predict(
                    frame,
                    imgsz=IMAGE_SIZE,
                    conf=self.confidence,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                    device="cpu"
                )

                result = results[0]

                detections = []

                current_time = time.time()

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )

                        confidence_score = float(
                            box.conf[0].item()
                        )

                        class_name = str(
                            class_names.get(
                                class_id,
                                f"Object {class_id}"
                            )
                        )

                        x1, y1, x2, y2 = (
                            box.xyxy[0].tolist()
                        )

                        center_x = (x1 + x2) / 2

                        frame_width = frame.shape[1]

                        if center_x < frame_width / 3:
                            position = "left"

                        elif center_x > (frame_width * 2 / 3):
                            position = "right"

                        else:
                            position = "centre"

                        detection = {
                            "class_id": class_id,
                            "name": class_name,
                            "confidence": confidence_score,
                            "position": position,
                            "box": (
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2)
                            )
                        }

                        detections.append(detection)

                        self.total_detections += 1

                        # Ignore selected classes for speech only.
                        # They remain visible in the detection list.

                        if class_name.lower().strip() in SILENT_CLASSES:
                            continue

                        key = (
                            class_id,
                            position
                        )

                        self.stability[key].append(True)

                        stable_count = sum(
                            self.stability[key]
                        )

                        if stable_count < STABILITY_REQUIRED:
                            continue

                        last_spoken = (
                            self.announcement_times.get(
                                key,
                                0
                            )
                        )

                        if (
                            current_time - last_spoken
                            >= VOICE_COOLDOWN
                        ):

                            message = (
                                f"{class_name} detected "
                                f"on your {position}"
                            )

                            speech_item = {
                                "id": str(uuid.uuid4()),
                                "message": message
                            }

                            self.pending_speech.append(
                                speech_item
                            )

                            self.last_announcement = message

                            self.announcement_times[key] = (
                                current_time
                            )

                with self.lock:

                    self.latest_detections = detections

                    self.last_time = current_time

                self.error = None

            except Exception as error:

                self.error = str(error)

                time.sleep(0.5)

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    def snapshot(self):

        with self.lock:

            detections = list(
                self.latest_detections
            )

            announcement = (
                self.last_announcement
            )

            speech_items = list(
                self.pending_speech
            )

            count = len(detections)

            last_time = self.last_time

            error = self.error

        return {
            "detections": detections,
            "announcement": announcement,
            "pending_speech": speech_items,
            "count": count,
            "last_time": last_time,
            "error": error
        }


# ============================================================
# BROWSER VOICE ANNOUNCEMENTS
# ============================================================

def speak_in_browser(items):

    if not items:
        return

    payload = json.dumps(items)

    components.html(
        f"""
        <script>

        const items = {payload};

        if (!window.visionGuideQueue) {{
            window.visionGuideQueue = [];
        }}

        if (!window.visionGuideQueuedIds) {{
            window.visionGuideQueuedIds = new Set();
        }}

        if (!window.visionGuideSpokenIds) {{
            window.visionGuideSpokenIds = new Set();
        }}

        items.forEach(item => {{

            if (
                !window.visionGuideQueuedIds.has(item.id) &&
                !window.visionGuideSpokenIds.has(item.id)
            ) {{

                window.visionGuideQueue.push(item);

                window.visionGuideQueuedIds.add(item.id);

            }}

        }});

        function processVisionGuideQueue() {{

            if (window.visionGuideActive) return;

            if (window.visionGuideQueue.length === 0) return;

            if (!window.speechSynthesis) return;

            const item = window.visionGuideQueue.shift();

            window.visionGuideActive = true;

            const utterance = new SpeechSynthesisUtterance(
                item.message
            );

            utterance.rate = 1.0;

            utterance.pitch = 1.0;

            utterance.volume = 1.0;

            let finished = false;

            function finish() {{

                if (finished) return;

                finished = true;

                window.visionGuideActive = false;

                window.visionGuideQueuedIds.delete(item.id);

                window.visionGuideSpokenIds.add(item.id);

                if (window.visionGuideSpokenIds.size > 300) {{

                    const first = window.visionGuideSpokenIds
                        .values().next().value;

                    window.visionGuideSpokenIds.delete(first);

                }}

                setTimeout(
                    processVisionGuideQueue,
                    300
                );

            }}

            utterance.onend = finish;

            utterance.onerror = finish;

            window.speechSynthesis.speak(
                utterance
            );

            setTimeout(
                finish,
                20000
            );

        }}

        processVisionGuideQueue();

        </script>
        """,
        height=0
    )


# ============================================================
# CAMERA
# ============================================================

st.markdown("## 📷 Live Camera Detection")

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
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

st.caption(
    "Allow camera permission in your browser to start detection."
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

if ctx.video_processor:

    @st.fragment(run_every=1.0)
    def live_dashboard():

        processor = ctx.video_processor

        data = processor.snapshot()

        detections = data["detections"]

        announcement = data["announcement"]

        speech_items = data["pending_speech"]

        if voice_enabled:
            speak_in_browser(speech_items)

        if data["error"]:

            st.error(
                f"Detection error: {data['error']}"
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Objects detected",
                data["count"]
            )

        with col2:

            st.metric(
                "Model classes",
                len(class_names)
            )

        with col3:

            st.metric(
                "Voice announcements",
                len(speech_items)
            )

        st.markdown("---")

        # ----------------------------------------------------
        # LAST ANNOUNCEMENT
        # ----------------------------------------------------

        st.markdown("### 🔊 Latest announcement")

        st.markdown(
            f"""
            <div class="vg-card">
                <h3>{announcement}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # DETECTED OBJECTS
        # ----------------------------------------------------

        st.markdown("### 🎯 Detected objects")

        if detections:

            for detection in detections:

                name = detection["name"]

                position = detection["position"]

                score = detection["confidence"]

                st.markdown(
                    f"""
                    <div class="vg-object">
                        <b>{name}</b><br>
                        Position: {position}<br>
                        Confidence: {score:.0%}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No objects detected yet. "
                "Keep the camera pointed towards the area."
            )

    live_dashboard()

else:

    st.info(
        "Start the camera to activate VisionGuide AI."
    )


# ============================================================
# MODEL CLASSES
# ============================================================

with st.expander("🧠 View trained model classes"):

    st.caption(
        "Class IDs and names are read directly from weights.pt."
    )

    columns = st.columns(3)

    for index, (class_id, class_name) in enumerate(
        class_names.items()
    ):

        with columns[index % 3]:

            st.write(
                f"**{class_id}** · {class_name}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#9ca3af;
        padding:20px;
        border-top:1px solid #263247;
    ">
        <b>VisionGuide AI</b><br>
        Experimental AI-powered object recognition prototype.
        <br><br>
        This prototype does not measure distance or guarantee
        obstacle avoidance. It must not be used as a replacement
        for a mobility aid or human assistance.
    </div>
    """,
    unsafe_allow_html=True
)
