
import streamlit as st
import streamlit.components.v1 as components
import cv2
import av
import time
import json
import threading

from collections import deque
from huggingface_hub import hf_hub_download
from rfdetr import RFDETRMedium
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VisionGuide AI",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ VisionGuide AI")
st.caption("AI-powered object detection and voice assistant")

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

HF_REPO_ID = "ommppra/visionguide-rfdetr"
HF_FILENAME = "weights.pt"

CONFIDENCE_THRESHOLD = 0.45
ANNOUNCEMENT_INTERVAL = 5


# --------------------------------------------------
# LOAD YOUR TRAINED RF-DETR MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading VisionGuide AI model...")
def load_model():

    hf_token = st.secrets["HF_TOKEN"]

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=hf_token,
    )

    loaded_model = RFDETRMedium(
        pretrain_weights=checkpoint_path
    )

    return loaded_model


try:
    model = load_model()
    st.success("RF-DETR Medium model loaded successfully.")

except Exception as error:
    st.error("Model loading failed.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# DISPLAY MODEL CLASS NAMES
# --------------------------------------------------

with st.expander("View model class names"):

    st.write(model.class_names)


# --------------------------------------------------
# CAMERA PROCESSOR
# --------------------------------------------------

class VisionGuideProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.detections = []
        self.speech_queue = deque(maxlen=5)

        self.last_announced = {}
        self.last_inference = 0

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        current_time = time.time()

        # Limit processing frequency to reduce CPU load.
        if current_time - self.last_inference < 0.8:
            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24",
            )

        self.last_inference = current_time

        try:

            # RF-DETR expects RGB input.
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            results = model.predict(
                rgb_image,
                threshold=CONFIDENCE_THRESHOLD,
            )

            boxes = results.xyxy
            scores = results.confidence
            class_ids = results.class_id

            # RF-DETR supplies trained class names
            # with fine-tuned detection results.
            class_names = results.data.get("class_name")

            current_detections = []
            voice_messages = []

            for index in range(len(boxes)):

                x1, y1, x2, y2 = map(
                    int,
                    boxes[index],
                )

                confidence = float(scores[index])
                class_id = int(class_ids[index])

                # Get the actual class name from the
                # trained checkpoint's prediction.
                if class_names is not None:

                    name = str(class_names[index])

                else:

                    names = model.class_names

                    if class_id in names:
                        name = str(names[class_id])
                    elif str(class_id) in names:
                        name = str(names[str(class_id)])
                    else:
                        name = f"Class {class_id}"

                current_detections.append({
                    "name": name,
                    "confidence": confidence,
                })

                # Draw bounding box.
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = f"{name} {confidence:.0%}"

                cv2.putText(
                    image,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Avoid repeating the same announcement
                # too frequently.
                previous_time = self.last_announced.get(
                    name,
                    0,
                )

                if (
                    current_time - previous_time
                    >= ANNOUNCEMENT_INTERVAL
                ):

                    voice_messages.append(
                        f"{name} detected."
                    )

                    self.last_announced[name] = current_time

            with self.lock:

                self.detections = current_detections

                if voice_messages:
                    self.speech_queue.append(
                        " ".join(voice_messages[:2])
                    )

        except Exception as error:

            with self.lock:
                self.detections = [{
                    "name": f"Detection error: {str(error)[:100]}",
                    "confidence": 0,
                }]

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def get_detections(self):

        with self.lock:
            return list(self.detections)

    def get_speech(self):

        with self.lock:

            if self.speech_queue:
                return self.speech_queue.popleft()

            return None


# --------------------------------------------------
# CAMERA INTERFACE
# --------------------------------------------------

st.divider()

camera_column, results_column = st.columns(
    [2, 1],
    gap="large",
)

with camera_column:

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
        "Allow camera access and press Start to begin detection."
    )


# --------------------------------------------------
# LIVE DETECTION DASHBOARD
# --------------------------------------------------

@st.fragment(run_every=1)
def update_dashboard():

    with results_column:

        st.subheader("🔍 Detected Objects")

        if not ctx.state.playing:

            st.info("Start the camera to begin.")

            return

        if ctx.video_processor is None:

            st.info("Starting camera processor...")

            return

        detections = (
            ctx.video_processor.get_detections()
        )

        if detections:

            st.metric(
                "Objects detected",
                len(detections),
            )

            for item in detections:

                st.write(
                    f"**{item['name']}**"
                    f" · {item['confidence']:.0%}"
                )

        else:

            st.info("No objects detected yet.")

        # Browser voice announcement.
        message = ctx.video_processor.get_speech()

        if message:

            safe_message = json.dumps(message)

            components.html(
                f"""
                <script>
                    const message = {safe_message};

                    if ("speechSynthesis" in window) {{
                        window.speechSynthesis.cancel();

                        const speech =
                            new SpeechSynthesisUtterance(message);

                        speech.rate = 1;
                        speech.pitch = 1;
                        speech.volume = 1;

                        window.speechSynthesis.speak(speech);
                    }}
                </script>
                """,
                height=0,
            )


update_dashboard()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "VisionGuide AI | RF-DETR Medium | "
    "Experimental demonstration"
)

st.warning(
    "This prototype can miss or misidentify objects. "
    "Do not rely on it for navigation, road crossing, "
    "stairs, or avoiding hazards."
)
