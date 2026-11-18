import streamlit as st
import cv2
import av
import time
import threading
from collections import deque
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

MODEL_PATH = "yolo26s.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.25
VOICE_COOLDOWN = 6

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

TARGET_OBJECTS = set(model.names.values())

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
    # ANNOUNCEMENT SYSTEM
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

            previous_time = self.last_announced.get(
                key, 0
            )

            if now - previous_time >= VOICE_COOLDOWN:

                if position == "centre":
                    direction = "ahead"
                else:
                    direction = "to your " + position

                if confidence > 0.80:

                    article = (
                        "an" if name[0].lower() in "aeiou"
                        else "a"
                    )

                    message = (
                        f"{article.capitalize()} {name} "
                        f"is {direction}."
                    )

                else:

                    message = (
                        f"There appears to be a {name} "
                        f"{direction}."
                    )

                announcements.append(message)
                announced_keys.add(key)

            if len(announcements) >= 4:
                break

        if announcements:

            message = " ".join(announcements)

            with self.lock:

                self.speech_queue.append(message)

            for key in announced_keys:

                self.last_announced[key] = now

    # ==================================================
    # CAMERA FRAME PROCESSING
    # ==================================================

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        # Process every second frame
        if self.frame_count % 2 == 0:

            results = model.predict(
                img,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                verbose=False
            )

            detections = []

            for result in results:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    name = model.names[class_id]

                    if name not in TARGET_OBJECTS:
                        continue

                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    center = (x1 + x2) / 2

                    if center < img.shape[1] / 3:

                        position = "left"

                    elif center > img.shape[1] * 2 / 3:

                        position = "right"

                    else:

                        position = "centre"

                    detections.append({

                        "name": name,

                        "position": position,

                        "confidence": round(
                            confidence, 2
                        )

                    })

                    # Draw bounding box
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{name} "
                        f"{confidence:.2f} "
                        f"- {position}"
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            # Update detections
            with self.lock:

                self.detections = detections

            # Voice announcements
            if detections:

                self.announce(detections)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
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

# ==================================================
# DASHBOARD
# ==================================================

left, right = st.columns(
    [2, 1],
    gap="large"
)

# ==================================================
# LEFT SIDE: CAMERA
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
            "audio": False
        },

        async_processing=True
    )

    st.caption(
        "Click Start to activate your camera "
        "and begin object detection."
    )

# ==================================================
# RIGHT SIDE: DASHBOARD
# ==================================================

@st.fragment(run_every=0.5)
def dashboard():

    with right:

        st.subheader("Controls")

        if ctx.state.playing:

            st.success("Detection is running")

        else:

            st.warning("Detection is stopped")

        st.caption(
            "Use the camera Start/Stop button "
            "to control detection."
        )

        st.divider()

        st.subheader("🔊 Voice Assistant")

        st.write("Voice announcements: Enabled")

        st.caption(
            "Browser speech synthesis"
        )

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

                st.caption(
                    "No objects detected yet."
                )

            # Speech announcement
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
                    height=0
                )

        else:

            st.caption(
                "Start the camera to detect objects."
            )

        st.divider()

        st.caption(
            f"Supported classes: "
            f"{len(TARGET_OBJECTS)}"
        )

        st.caption(
            f"Detection image size: {IMAGE_SIZE}"
        )

dashboard()

st.divider()

st.caption(
    "VisionGuide AI is an experimental object "
    "detection assistant. Do not rely on it "
    "for safe navigation."
)
