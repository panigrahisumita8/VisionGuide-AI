
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

st.set_page_config(page_title="VisionGuide AI", page_icon="👁️", layout="wide")


# ============================================================
# FIREBASE / GOOGLE ANALYTICS
# ============================================================

FIREBASE_MEASUREMENT_ID = "G-XSHL6SPZ9G"

st.html(
    f"""
    <script>
    (function () {{
        if (window.__visionguideFirebaseAnalyticsLoaded) return;

        window.__visionguideFirebaseAnalyticsLoaded = true;

        var measurementId = "{FIREBASE_MEASUREMENT_ID}";
        var script = document.createElement("script");

        script.async = true;
        script.src =
            "https://www.googletagmanager.com/gtag/js?id=" + measurementId;

        document.head.appendChild(script);

        window.dataLayer = window.dataLayer || [];

        function gtag() {{
            window.dataLayer.push(arguments);
        }}

        window.gtag = gtag;

        gtag("js", new Date());

        gtag("config", measurementId, {{
            send_page_view: true,
            page_title: "VisionGuide AI",
            page_location: window.top.location.href
        }});
    }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)


# ============================================================
# CUSTOM DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {background:#0b1020;color:#f1f5f9}
[data-testid="stSidebar"] {background:#111827}
.vg-title {font-size:2.1rem;font-weight:800;color:#f8fafc;margin-bottom:0}
.vg-subtitle {color:#94a3b8;margin-top:.2rem;margin-bottom:1.2rem}
div[data-testid="stMetric"] {background:#172033;border:1px solid #29354c;padding:14px;border-radius:12px}
div[data-testid="stVerticalBlockBorderWrapper"] {background:#111a2b;border:1px solid #29354c;border-radius:12px}
.stButton button {border-radius:9px}
hr {border-color:#29354c}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="vg-title">👁️ VisionGuide AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="vg-subtitle">AI-powered object detection assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = os.getenv("VISIONGUIDE_MODEL", "weights.pt")
IMAGE_SIZE = 416
DEFAULT_CONFIDENCE = 0.35
VOICE_COOLDOWN = 7
STABILITY_REQUIRED = 2
MAX_DETECTIONS = 25


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 👁️ VisionGuide")
    st.caption("AI Vision Assistant")
    st.divider()

    st.markdown("### Detection settings")

    confidence_setting = st.slider(
        "Detection confidence",
        0.15,
        0.90,
        DEFAULT_CONFIDENCE,
        0.05
    )

    voice_enabled = st.toggle(
        "Voice announcements",
        value=True
    )

    st.divider()

    st.markdown("### Model information")
    st.caption(f"Checkpoint: `{MODEL_PATH}`")
    st.caption(f"Inference image size: {IMAGE_SIZE}px")
    st.caption("Background AI processing")

    st.divider()
    st.caption("VisionGuide AI | Experimental prototype")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading trained VisionGuide model...")
def load_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"'{model_path}' was not found. Keep weights.pt beside streamlit_app.py."
        )

    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH)

    class_names = model.names

    if isinstance(class_names, list):
        class_names = dict(enumerate(class_names))

except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# ============================================================
# VISIONGUIDE CAMERA PROCESSOR
# ============================================================

class VisionGuideProcessor:
    """Keeps camera output live while a worker performs YOLO inference."""

    def __init__(self, yolo_model, confidence_value):
        self.model = yolo_model
        self.confidence = confidence_value

        self.lock = threading.Lock()
        self.queue_lock = threading.Lock()

        self.latest_frame = None
        self.boxes_for_display = []
        self.detections = []

        self.last_announcement = ""
        self.announcement_id = ""

        self.last_announced_at = {}

        self.stability = defaultdict(lambda: deque(maxlen=4))

        self.error = ""
        self.inference_count = 0
        self.last_inference_time = 0.0

        self.worker = threading.Thread(
            target=self.detection_worker,
            daemon=True
        )

        self.worker.start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        image = frame.to_ndarray(format="bgr24")

        with self.queue_lock:
            self.latest_frame = image.copy()

        output = image.copy()

        with self.lock:
            boxes = list(self.boxes_for_display)

        h, w = output.shape[:2]

        for item in boxes:

            x1, y1, x2, y2 = item["box"]

            sx = w / max(1, item["frame_width"])
            sy = h / max(1, item["frame_height"])

            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 220, 130),
                2
            )

            cv2.putText(
                output,
                f'{item["name"]} {item["confidence"]:.0%}',
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 130),
                2
            )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

    def detection_worker(self):

        while True:

            with self.queue_lock:
                image = self.latest_frame
                self.latest_frame = None

            if image is None:
                time.sleep(0.015)
                continue

            try:

                height, width = image.shape[:2]

                results = self.model.predict(
                    image,
                    imgsz=IMAGE_SIZE,
                    conf=self.confidence,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                    device="cpu"
                )

                current_detections = []
                current_boxes = []
                present_keys = set()

                for result in results:

                    if result.boxes is None:
                        continue

                    for box in result.boxes:

                        class_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())

                        name = str(self.model.names[class_id])

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        center_x = (x1 + x2) / 2

                        if center_x < width / 3:
                            position = "left"

                        elif center_x > width * 2 / 3:
                            position = "right"

                        else:
                            position = "centre"

                        current_detections.append({
                            "name": name,
                            "class_id": class_id,
                            "position": position,
                            "confidence": score
                        })

                        current_boxes.append({
                            "name": name,
                            "confidence": score,
                            "box": (x1, y1, x2, y2),
                            "frame_width": width,
                            "frame_height": height
                        })

                        present_keys.add((class_id, position))

                now = time.time()

                all_keys = set(self.stability.keys()) | present_keys

                for key in all_keys:

                    self.stability[key].append(
                        1 if key in present_keys else 0
                    )

                announcements = []

                for class_id, position in sorted(present_keys):

                    key = (class_id, position)

                    if sum(self.stability[key]) < STABILITY_REQUIRED:
                        continue

                    if now - self.last_announced_at.get(key, 0) < VOICE_COOLDOWN:
                        continue

                    name = str(self.model.names[class_id])

                    announcements.append(
                        f"{name} detected on the {position}"
                    )

                    self.last_announced_at[key] = now

                announcements = list(
                    dict.fromkeys(announcements)
                )[:4]

                with self.lock:

                    self.detections = current_detections
                    self.boxes_for_display = current_boxes

                    self.error = ""

                    self.inference_count += 1
                    self.last_inference_time = time.time()

                    if announcements:

                        self.last_announcement = ". ".join(
                            announcements
                        ) + "."

                        self.announcement_id = str(uuid.uuid4())

            except Exception as exc:

                with self.lock:
                    self.error = str(exc)

    def snapshot(self):

        with self.lock:

            return (
                list(self.detections),
                self.last_announcement,
                self.announcement_id,
                self.error,
                self.inference_count,
                self.last_inference_time
            )


# ============================================================
# BROWSER VOICE ANNOUNCEMENTS
# ============================================================

def speak_in_browser(message: str, event_id: str):

    html = f"""
    <script>
    const message = {json.dumps(message)};
    const eventId = {json.dumps(event_id)};

    if (message && eventId && "speechSynthesis" in window) {{

        const key = "visionguide_last_spoken_event";
        let previous = "";

        try {{
            previous = window.sessionStorage.getItem(key) || "";
        }} catch(e) {{}}

        if (eventId !== previous) {{

            try {{
                window.sessionStorage.setItem(key, eventId);
            }} catch(e) {{}}

            const utterance = new SpeechSynthesisUtterance(message);

            utterance.lang = "en-US";
            utterance.rate = 1;
            utterance.pitch = 1;

            window.speechSynthesis.speak(utterance);
        }}
    }}
    </script>
    """

    components.html(
        html,
        height=0,
        width=0
    )


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

    video_processor_factory=lambda: VisionGuideProcessor(
        model,
        confidence_setting
    ),

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

st.caption(
    "Allow camera permission in your browser. Use Start to begin and Stop to pause."
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(run_every=1.0)
def live_dashboard():

    left, right = st.columns(
        [2, 1],
        gap="large"
    )

    processor = (
        ctx.video_processor
        if ctx.state.playing
        else None
    )

    with left:

        st.subheader("Detection Preview")

        if ctx.state.playing:

            st.success("Camera connected")

            st.caption(
                "The annotated live feed appears in the camera panel above."
            )

        else:

            st.info(
                "Camera is stopped. Start it in the panel above."
            )

    with right:

        st.subheader("System Status")

        if processor is not None:

            st.success(
                "VisionGuide is processing camera frames"
            )

        elif ctx.state.signalling:

            st.info("Connecting to camera...")

        else:

            st.warning("Camera is not running")

        st.divider()

        st.subheader("🔊 Voice Assistant")

        st.write(
            "Browser voice announcements: "
            + ("Enabled" if voice_enabled else "Disabled")
        )

        st.caption(
            "Uses the visitor's browser speech engine."
        )

    st.divider()

    if processor is None:

        st.info("No active camera stream.")
        return

    (
        detections,
        announcement,
        announcement_id,
        error,
        count,
        last_time
    ) = processor.snapshot()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Objects detected",
        len(detections)
    )

    col2.metric(
        "AI frames processed",
        count
    )

    col3.metric(
        "Last AI update",
        f"{time.time() - last_time:.1f}s ago"
        if last_time
        else "Waiting"
    )

    if error:
        st.error(f"Detection error: {error}")

    st.subheader("🔊 Latest Announcement")

    if announcement:

        st.info(announcement)

        if voice_enabled:
            speak_in_browser(
                announcement,
                announcement_id
            )

    else:

        st.caption(
            "Waiting for stable object detections."
        )

    st.subheader("🎯 Detected Objects")

    if detections:

        best_by_class = {}

        for item in detections:

            key = (
                item["class_id"],
                item["position"]
            )

            if (
                key not in best_by_class
                or item["confidence"] > best_by_class[key]["confidence"]
            ):

                best_by_class[key] = item

        for item in sorted(
            best_by_class.values(),
            key=lambda x: x["confidence"],
            reverse=True
        ):

            st.write(
                f'**{item["name"]}** · {item["position"].title()} '
                f'· {item["confidence"]:.0%} · ID {item["class_id"]}'
            )

    else:

        st.caption(
            "No objects detected in the latest processed frame."
        )


live_dashboard()


# ============================================================
# TRAINED MODEL CLASSES
# ============================================================

st.divider()

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

st.divider()

st.caption(
    "VisionGuide AI is an experimental educational prototype. It may make "
    "incorrect detections and cannot measure distance or guarantee obstacle "
    "avoidance. Do not rely on it as a substitute for a mobility aid or human assistance."
)
