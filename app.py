import cv2
import numpy as np
import streamlit as st
import time
from detection import initialize_detection, process_frame
import torch
import os

# Optional: Prevent JIT profiling issues in some environments
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Set page layout
st.set_page_config(layout="wide")
st.title("🚗 Real-Time Parking Slot Detection ")

# Styling for metric boxes
st.markdown("""
<style>
    .stMetric {
        background-color: #D3D3D3;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Video frame width - keep consistent with cv2.resize below
VIDEO_WIDTH = 1020
VIDEO_HEIGHT = 500

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = True
    st.session_state.pause = False
    st.session_state.show_cv2 = True
    st.session_state.alert_state = "normal"

# Layout columns for video and controls
col_video, col_controls = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    status_text = st.empty()
    available_slots_placeholder = st.empty()

    # Alert placeholder for horizontal alert bar (fixed width = VIDEO_WIDTH)
    alert_bar_placeholder = st.empty()

with col_controls:
    st.header("🎛️ Controls")
    st.session_state.show_cv2 = st.checkbox("Show OpenCV Window", value=True)

    if st.button("⏸️ Pause/Resume"):
        st.session_state.pause = not st.session_state.pause

    if st.button("⏹️ Stop Processing"):
        st.session_state.running = False

    st.markdown("---")
    st.subheader("📊 Parking Slot Stats")
    total_slots_placeholder = st.empty()
    occupied_placeholder = st.empty()
    available_placeholder = st.empty()
    st.subheader("🚨 Alerts")
    alert_placeholder = st.empty()

    st.subheader("🎥 Frame Info")
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    st.markdown("---")
    st.info("ℹ️ Detection running in real-time")

# Load model and slot polygons
model, class_list, area_polygons = initialize_detection()
if model is None:
    st.error("❌ Failed to initialize model")
    st.stop()

# Open video
cap = cv2.VideoCapture('parking1.mp4')
if not cap.isOpened():
    st.error("❌ Could not open video file")
    st.stop()

frame_count = 0
start_time = time.time()
total_slots = len(area_polygons)

# HTML + CSS for red horizontal scrolling alert bar with fixed width = VIDEO_WIDTH
def get_alert_bar_html():
    return f"""
    <style>

    .alert-container {{
      width: 100%;
      display: flex;
      justify-content: center;
    }}
    .marquee {{
      max-width: 100%;
      overflow: hidden;
      background-color: red;
      color: white;
      font-weight: bold;
      font-size: 22px;
      white-space: nowrap;
      box-sizing: border-box;
      padding: 12px 0;
      margin: 10px auto 0 auto;
      border-radius: 6px 6px 0 0;
      box-shadow: 0 0 10px rgba(255, 0, 0, 0.7);
      position: relative;
      text-align: center;
    }}

    </style>
    <div class="alert-container">
      <div class="marquee">
        <span>🚨 NO MORE PARKING SPACE AVAILABLE! PLEASE WAIT 🚨</span>
      </div>
    </div>
    """


while st.session_state.running and cap.isOpened():
    if not st.session_state.pause:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("⚠️ Video ended")
            break

        frame_count += 1
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        frame, occupancy = process_frame(frame, model, class_list, area_polygons)

        occupied = sum(occupancy)
        available = total_slots - occupied
        occupancy_percentage = occupied / total_slots * 100
        current_fps = frame_count / (time.time() - start_time)

        # Update metrics
        total_slots_placeholder.metric("Total Slots", total_slots)
        occupied_placeholder.metric("Occupied Slots", occupied)
        available_placeholder.metric("Available Slots", available)
        frame_placeholder.metric("Frame Count", frame_count)
        fps_placeholder.metric("FPS", f"{current_fps:.2f}")

        # Available slots display (colored badges)
        available_slot_nums = [f"Slot {i+1}" for i, occ in enumerate(occupancy) if occ == 0]
        if available_slot_nums:
            if len(available_slot_nums) <= 4:
                slots_html = "".join([
                    f"<span style='background-color:#f8d7da; color:#721c24; padding:6px 12px; "
                    f"margin:4px; border-radius:8px; display:inline-block; font-weight:bold;'>"
                    f"{slot}</span>"
                    for slot in available_slot_nums
                ])
                available_slots_placeholder.markdown(
                    f"""
                    <h4>🅿️ <u>Available Slots:</u></h4>
                    <div style='margin-top:10px'>{slots_html}</div>
                    <div style='margin-top:10px; color:red; font-weight:bold; font-size:16px'>
                        🚨 Only 33% Parking slots are left!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                slots_html = "".join([
                    f"<span style='background-color:#d4edda; color:#155724; padding:6px 12px; "
                    f"margin:4px; border-radius:8px; display:inline-block; font-weight:bold;'>"
                    f"{slot}</span>"
                    for slot in available_slot_nums
                ])
                available_slots_placeholder.markdown(
                    f"<h4>🅿️ <u>Available Slots:</u></h4><div style='margin-top:10px'>{slots_html}</div>",
                    unsafe_allow_html=True
                )
        else:
            available_slots_placeholder.markdown(
                "<h4>🅿️ <u>No Available Slots</u></h4>",
                unsafe_allow_html=True
            )

        # Alerts in normal alert box area (above alert bar)
        if occupancy_percentage >= 100:
            if st.session_state.alert_state != "full":
                alert_placeholder.error("🚨 Parking Full: No more parking space available!")
                st.session_state.alert_state = "full"
        elif occupancy_percentage >= 80:
            if st.session_state.alert_state != "warning":
                alert_placeholder.warning("⚠️ High Occupancy Alert: Few parking slots left!")
                st.session_state.alert_state = "warning"
        else:
            if st.session_state.alert_state != "normal":
                alert_placeholder.empty()
                st.session_state.alert_state = "normal"

        # Show red horizontal marquee alert bar only if parking full (available == 0)
        if available == 0:
            alert_bar_placeholder.markdown(get_alert_bar_html(), unsafe_allow_html=True)
        else:
            alert_bar_placeholder.empty()

        # Draw polygons and labels on frame
        for i in range(total_slots):
            color = (0, 0, 255) if occupancy[i] else (0, 255, 0)
            cv2.polylines(frame, [area_polygons[i]], True, color, 2)

            top_left_idx = np.argmin(np.sum(area_polygons[i], axis=1))
            top_left_point = area_polygons[i][top_left_idx]
            label_x = max(top_left_point[0] - 10, 0)
            label_y = max(top_left_point[1] - 10, 0)
            label_position = (label_x, label_y)

            cv2.putText(frame, f"Slot {i+1}", label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Available: {available}", (23, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Show OpenCV window if enabled
        if st.session_state.show_cv2 and os.environ.get("DISPLAY"):
            cv2.imshow('Parking Detection', frame)

        # Show frame in Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        status_text.success(f"▶️ Frame {frame_count} | FPS: {current_fps:.1f}")

        # Exit on ESC in OpenCV window
        if os.environ.get("DISPLAY") and (cv2.waitKey(1) & 0xFF == 27):
            st.session_state.running = False
            break
    else:
        status_text.warning("⏸️ Processing Paused")
        time.sleep(0.2)

# Clean up
cap.release()
if os.environ.get("DISPLAY"):
    cv2.destroyAllWindows()

st.success("🛑 Processing stopped.")