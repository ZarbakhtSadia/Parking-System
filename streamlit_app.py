import cv2
import pandas as pd
import numpy as np
import streamlit as st
import time
from PIL import Image
from detection import initialize_detection, process_frame
import torch
import os

# Optional: Prevent JIT profiling issues in some environments
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Set page layout
st.set_page_config(layout="wide")
st.title("🚗 Real-Time Parking Slot Detection (Default Video)")

# Styling
st.markdown("""
<style>
    .stMetric {
        background-color: #D3D3D3;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = True
    st.session_state.pause = False
    st.session_state.show_cv2 = True
    st.session_state.alert_state = "normal"  # can be "normal", "warning", or "full"

# Layout
col_video, col_controls = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    status_text = st.empty()

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
    alert_placeholder = st.empty()  # Placeholder for alerts

    st.subheader("🎥 Frame Info")
    frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    st.markdown("---")
    st.info("ℹ️ Detection running in real-time")

# Load YOLO model and predefined slot polygons
model, class_list, area_polygons = initialize_detection()

if model is None:
    st.error("❌ Failed to initialize model")
    st.stop()

# Open default video
cap = cv2.VideoCapture('parking1.mp4')
if not cap.isOpened():
    st.error("❌ Could not open video file")
    st.stop()

# Optional mouse position debug
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse: {x}, {y}")

# # Setup OpenCV window
# Only create OpenCV window if display is available and user wants it
if st.session_state.show_cv2 and os.environ.get("DISPLAY"):
    cv2.namedWindow('Parking Detection')
    cv2.setMouseCallback('Parking Detection', mouse_callback)

# Start processing loop
frame_count = 0
start_time = time.time()
total_slots = len(area_polygons)

while st.session_state.running and cap.isOpened():
    if not st.session_state.pause:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("⚠️ Video ended")
            break

        frame_count += 1
        frame = cv2.resize(frame, (1020, 500))

        # Detection
        frame, occupancy = process_frame(frame, model, class_list, area_polygons)

        # Calculate stats
        occupied = sum(occupancy)
        available = total_slots - occupied
        occupancy_percentage = occupied / total_slots * 100
        current_fps = frame_count / (time.time() - start_time)

        # Update dashboard metrics
        total_slots_placeholder.metric("Total Slots", total_slots)
        occupied_placeholder.metric("Occupied Slots", occupied)
        available_placeholder.metric("Available Slots", available)
        frame_placeholder.metric("Frame Count", frame_count)
        fps_placeholder.metric("FPS", f"{current_fps:.2f}")

        # Alert logic (only update when state changes)
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
                alert_placeholder.empty()  # Clear alert
                st.session_state.alert_state = "normal"

        # Draw polygons
        for i in range(total_slots):
            color = (0, 0, 255) if occupancy[i] else (0, 255, 0)
            cv2.polylines(frame, [area_polygons[i]], True, color, 2)

        # Draw available count
        cv2.putText(frame, str(available), (23, 30),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

        # Display in OpenCV window only if display is available
        if st.session_state.show_cv2 and os.environ.get("DISPLAY"):
            cv2.imshow('Parking Detection', frame)

        # Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        status_text.success(f"▶️ Frame {frame_count} | FPS: {current_fps:.1f}")

        # Handle keypress only if display is available
        if os.environ.get("DISPLAY") and (cv2.waitKey(1) & 0xFF == 27):
            st.session_state.running = False
            break
    else:
        status_text.warning("⏸️ Paused")
        time.sleep(0.1)

# Cleanup
cap.release()
if os.environ.get("DISPLAY"):
    cv2.destroyAllWindows()
status_text.success("✅ Processing completed!")
