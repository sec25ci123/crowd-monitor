
import streamlit as st
import cv2
import av
import os
import time
import tempfile
import threading
import requests

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="SentraPulse",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 SentraPulse")
st.subheader(
    "AI-Based Crowd Density Prediction "
    "and Safety Alert System"
)

st.caption(
    "Video Upload | Live Camera | AI People Detection"
)

# -----------------------------
# LOAD AI MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# -----------------------------
# CROWD RISK CALCULATION
# -----------------------------

def calculate_risk(count, area):

    density = count / area

    if density < 2:
        risk = "LOW"

    elif density < 3:
        risk = "MODERATE"

    elif density < 4:
        risk = "HIGH"

    else:
        risk = "CRITICAL"

    return density, risk


def get_alert(risk):

    messages = {
        "LOW": "Normal crowd. Continue monitoring.",
        "MODERATE": "Crowd increasing. Monitor the area.",
        "HIGH": "Warning! High crowd density detected.",
        "CRITICAL": "Critical crowd density! Immediate safety action required."
    }

    return messages[risk]


# -----------------------------
# TELEGRAM ALERT
# -----------------------------

def send_telegram_alert(message):

    try:
        config = st.secrets.get("telegram", {})
        token = config.get("bot_token", "")
        chat_id = config.get("chat_id", "")

        if not token or not chat_id:
            return False

        url = (
            f"https://api.telegram.org/"
            f"bot{token}/sendMessage"
        )

        response = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": "🚨 SENTRAPULSE ALERT\n\n" + message
            },
            timeout=5
        )

        return response.ok

    except Exception:
        return False


# -----------------------------
# AI DETECTION
# -----------------------------

def detect_people(model, frame):

    result = model.predict(
        frame,
        classes=[0],
        conf=0.35,
        imgsz=640,
        verbose=False
    )[0]

    count = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )

    annotated = result.plot()

    return count, annotated


# -----------------------------
# SHARED LIVE CAMERA DATA
# -----------------------------

class LiveState:

    def __init__(self):

        self.lock = threading.Lock()
        self.count = 0
        self.density = 0.0
        self.risk = "LOW"
        self.previous_alert = None


def process_live_frame(frame, model, state, area):

    image = frame.to_ndarray(format="bgr24")

    count, annotated = detect_people(model, image)

    density, risk = calculate_risk(count, area)

    with state.lock:

        state.count = count
        state.density = density
        state.risk = risk

    cv2.putText(
        annotated,
        f"People: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Density: {density:.2f} persons/m2",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"Risk: {risk}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    return av.VideoFrame.from_ndarray(
        annotated,
        format="bgr24"
    )


# -----------------------------
# LIVE DASHBOARD REFRESH
# -----------------------------

@st.fragment(run_every="1s")
def live_dashboard(state):

    with state.lock:

        count = state.count
        density = state.density
        risk = state.risk
        previous = state.previous_alert

    col1, col2, col3 = st.columns(3)

    col1.metric("People Detected", count)

    col2.metric(
        "Crowd Density",
        f"{density:.2f} /m²"
    )

    col3.metric("Risk Level", risk)

    st.info(get_alert(risk))

    # Send Telegram only when risk changes
    if risk in ["HIGH", "CRITICAL"] and risk != previous:

        send_telegram_alert(
            f"Risk Level: {risk}\n"
            f"People: {count}\n"
            f"Density: {density:.2f} persons/m2\n"
            f"Alert: {get_alert(risk)}"
        )

    with state.lock:
        state.previous_alert = risk


# -----------------------------
# INPUT SETTINGS
# -----------------------------

st.sidebar.header("Monitoring Settings")

area = st.sidebar.number_input(
    "Monitored area (square metres)",
    min_value=0.1,
    max_value=100000.0,
    value=50.0,
    step=5.0
)

st.sidebar.caption(
    "Enter the actual monitored floor area. "
    "Accurate density requires camera calibration."
)

mode = st.radio(
    "Choose monitoring method",
    ["Upload Video", "Live Camera"],
    horizontal=True
)


# -----------------------------
# VIDEO UPLOAD MODE
# -----------------------------

if mode == "Upload Video":

    st.header("📁 Upload Crowd Video")

    uploaded = st.file_uploader(
        "Choose a video from your phone",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded is not None:

        st.video(uploaded)

        if st.button("🚀 Analyse Crowd", type="primary"):

            model = load_model()

            suffix = os.path.splitext(
                uploaded.name
            )[1] or ".mp4"

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp:

                    temp.write(uploaded.getvalue())
                    temp_path = temp.name

                cap = cv2.VideoCapture(temp_path)

                total = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if not cap.isOpened():
                    st.error("Unable to open this video.")
                    st.stop()

                st.header("📊 Crowd Analysis")

                col1, col2, col3 = st.columns(3)

                count_box = col1.empty()
                density_box = col2.empty()
                risk_box = col3.empty()

                image_box = st.empty()
                progress = st.progress(0)

                frame_number = 0
                max_count = 0
                max_density = 0
                last_risk = None

                # Process every third frame
                while cap.isOpened():

                    success, frame = cap.read()

                    if not success:
                        break

                    frame_number += 1

                    if frame_number % 3 != 0:
                        continue

                    count, annotated = detect_people(
                        model,
                        frame
                    )

                    density, risk = calculate_risk(
                        count,
                        area
                    )

                    max_count = max(max_count, count)
                    max_density = max(
                        max_density,
                        density
                    )

                    count_box.metric(
                        "Current People",
                        count
                    )

                    density_box.metric(
                        "Current Density",
                        f"{density:.2f} /m²"
                    )

                    risk_box.metric(
                        "Current Risk",
                        risk
                    )

                    image_box.image(
                        cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        ),
                        channels="RGB",
                        use_container_width=True
                    )

                    if risk in ["HIGH", "CRITICAL"] and risk != last_risk:

                        send_telegram_alert(
                            f"Video Analysis Alert\n"
                            f"Risk: {risk}\n"
                            f"People: {count}\n"
                            f"Density: {density:.2f} persons/m2"
                        )

                    last_risk = risk

                    if total > 0:
                        progress.progress(
                            min(frame_number / total, 1.0)
                        )

                cap.release()
                progress.progress(1.0)

                st.success("Video analysis completed!")

                st.subheader("Maximum Observed Values")

                st.write(
                    f"Maximum people detected: {max_count}"
                )

                st.write(
                    f"Maximum estimated density: "
                    f"{max_density:.2f} persons/m²"
                )

            except Exception as e:

                st.error(f"Analysis error: {e}")

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# -----------------------------
# LIVE CAMERA MODE
# -----------------------------

elif mode == "Live Camera":

    st.header("📹 Real-Time Crowd Monitoring")

    camera = st.selectbox(
        "Select your mobile camera",
        ["Rear Camera", "Front Camera"]
    )

    facing = (
        "environment"
        if camera == "Rear Camera"
        else "user"
    )

    st.info(
        "Allow camera access when your browser asks "
        "for permission."
    )

    model = load_model()

    state = st.session_state.get("live_state")

    if state is None:
        state = LiveState()
        st.session_state.live_state = state

    ctx = webrtc_streamer(
        key=f"sentrapulse-{facing}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        media_stream_constraints={
            "video": {
                "facingMode": {
                    "ideal": facing
                }
            },
            "audio": False
        },
        video_frame_callback=lambda frame: process_live_frame(
            frame,
            model,
            state,
            area
        ),
        async_processing=True
    )

    live_dashboard(state)

    if not ctx.state.playing:
        st.caption(
            "Press START to begin live crowd monitoring."
        )

    st.caption(
        "Press STOP to end the live camera session."
    )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "SentraPulse | AI-Based Crowd Safety Monitoring"
)

st.caption(
    "Prototype only. Alerts and density estimates "
    "must be validated before real-world safety use."
              )
