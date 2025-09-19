# TRIDENT AI Vision System
# Copyright (C) 2025  Rejey O. Gammad
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import streamlit as st
import cv2
import tempfile
import time
import os
import torch
import mimetypes
import numpy as np
from datetime import datetime, timedelta

# Import your main pipeline - adjust path as needed
from src.main import VideoReIDPipeline

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="TRIDENT AI Vision System",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS for Professional Styling
# --------------------------
st.markdown("""
    <style>
        /* Overall background */
        .main {
            background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%);
            color: #f5f5f5;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* Title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            text-align: center;
            color: #a0a0a0;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1b1f2a 0%, #2d3748 100%);
            border-right: 1px solid #2d3748;
        }
        
        .sidebar-header {
            color: #667eea;
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
            
        /* Container width */
        .block-container {
            max-width: 95% !important;
            padding: 2rem 1rem;
        }

        /* Video container */
        .video-frame {
            background: #000;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Log box styling */
        .log-box {
            background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
            border: 1px solid #4a5568;
            border-radius: 12px;
            padding: 15px;
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            color: #48bb78;
            height: 500px;
            overflow-y: auto;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .log-entry {
            margin: 4px 0;
            padding: 2px 0;
            border-bottom: 1px solid rgba(74, 85, 104, 0.3);
        }

        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }

        /* Metrics styling */
        [data-testid="metric-container"] {
            background: rgba(26, 31, 46, 0.6);
            border: 1px solid #4a5568;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }

        /* Hints section */
        .hints-container {
            background: rgba(26, 31, 46, 0.8);
            border: 1px solid #4a5568;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }

        .hint-title {
            color: #667eea;
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }

        .hint-item {
            margin: 0.8rem 0;
            padding-left: 1.5rem;
            position: relative;
            color: #cbd5e0;
            line-height: 1.6;
        }

        .hint-item:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: #667eea;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #718096;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #2d3748;
        }

        /* Slider labels */
        .stSlider label {
            color: #e2e8f0 !important;
            font-weight: 500;
        }

        /* Toggle switch labels */
        label[data-baseweb="checkbox"] span {
            color: #e2e8f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Initialize Session State
# --------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "fps" not in st.session_state:
    st.session_state.fps = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "id_streaks" not in st.session_state:
    st.session_state.id_streaks = {}
if "confirmed_ids" not in st.session_state:
    st.session_state.confirmed_ids = set()
if "last_seen" not in st.session_state:
    st.session_state.last_seen = {}
if "last_bbox" not in st.session_state:
    st.session_state.last_bbox = {}
if "was_missing" not in st.session_state:
    st.session_state.was_missing = {}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "disabled_params" not in st.session_state:
    st.session_state.disabled_params = False
if "enable_stop_btn" not in st.session_state:
    st.session_state.enable_stop_btn = True
if "ema_off" not in st.session_state:
    st.session_state.ema_off = False
if "overlay_off" not in st.session_state:
    st.session_state.overlay_off = True

if "parameters" not in st.session_state:
    st.session_state.parameters = {
        "nms_threshold": 0.45,
        "iou_threshold": 0.3,
        "cosine_threshold": 0.65,
        "ema_alpha": 0.9,
        "proximity_threshold": 50,
    }
if "use_gpu" not in st.session_state:
    st.session_state.use_gpu = True

def update_nms():
    """Update NMS threshold in session state"""
    st.session_state.parameters["nms_threshold"] = st.session_state.nms_threshold

def update_iou():
    """Update IoU threshold in session state"""
    st.session_state.parameters["iou_threshold"] = st.session_state.iou_threshold

def update_cosine():
    """Update Cosine Similarity threshold in session state"""
    st.session_state.parameters["cosine_threshold"] = st.session_state.cosine_threshold

def update_ema():
    """Update EMA alpha in session state"""
    st.session_state.parameters["ema_alpha"] = st.session_state.ema_alpha

def update_proximity_th():
    """Update Proximity threshold in session state"""
    st.session_state.parameters["proximity_threshold"] = st.session_state.proximity_threshold

def params_disabled():
    """Check if the pipeline is currently processing"""
    st.session_state.disabled_params = not st.session_state.disabled_params
    st.session_state.enable_stop_btn = not st.session_state.enable_stop_btn

def disable_overlay():
    """Disable overlay display"""
    st.session_state.overlay_off = not st.session_state.overlay_off

def disable_ema():
    """Disable EMA updates"""
    st.session_state.ema_off = not st.session_state.ema_off

def disable_gpu():
    """Disable GPU usage"""
    st.session_state.use_gpu = not st.session_state.use_gpu

# --------------------------
# Main Title and Description
# --------------------------
st.markdown('<h1 class="main-title">🔱 TRIDENT AI Vision System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload your own video for testing - See beyond mAP, CMC, MOTA, and IDF1 numbers, experience it in action!</p>',
    unsafe_allow_html=True
)

# --------------------------
# Sidebar Configuration
# --------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration Panel</div>', unsafe_allow_html=True)
    
    # File Upload
    st.markdown("### 📁 Video Input")
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    
    if video_file:
        st.success(f"✅ Loaded: {video_file.name}")
    
    st.markdown("---")
    
    # Detection Parameters
    st.markdown("### 🎯 Detection Parameters")
    
    nms_threshold = st.slider(
        "NMS Threshold",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.parameters["nms_threshold"],
        step=0.05,
        key="nms_threshold",
        help="Non-Maximum Suppression threshold for filtering overlapping detections",
        disabled=st.session_state.disabled_params,
        on_change=update_nms
    )
    
    iou_threshold = st.slider(
        "IoU Threshold (Occlusion)",
        min_value=0.1,
        max_value=0.8,
        value=st.session_state.parameters["iou_threshold"],
        step=0.05,
        help="Intersection over Union threshold for occlusion detection",
        disabled=st.session_state.disabled_params,
        on_change=update_iou
    )
    
    st.markdown("---")
    
    # ReID Parameters
    st.markdown("### 🔍 Re-Identification Parameters")
    
    cosine_threshold = st.slider(
        "Cosine Similarity Threshold",
        min_value=0.3,
        max_value=0.95,
        value=st.session_state.parameters["cosine_threshold"],
        step=0.05,
        help="Minimum similarity score for matching identities",
        key="cosine_threshold",
        disabled=st.session_state.disabled_params,
        on_change=update_cosine
    )
    
    ema_alpha = st.slider(
        "EMA Alpha",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.parameters["ema_alpha"],
        step=0.05,
        key="ema_alpha",
        help="Exponential Moving Average weight for feature updates",
        disabled=st.session_state.ema_off or st.session_state.disabled_params,
        on_change=update_ema
    )
    
    proximity_threshold = st.slider(
        "Proximity Threshold (pixels)",
        min_value=10,
        max_value=200,
        value=st.session_state.parameters["proximity_threshold"],
        step=10,
        key="proximity_threshold",
        help="Maximum pixel distance for re-acquisition detection",
        disabled=st.session_state.disabled_params,
        on_change=update_proximity_th
    )
    
    st.markdown("---")
    
    # Display Options
    st.markdown("### 🎨 Display Options")
    
    show_overlay = st.toggle(
        "Show System Overlay",
        value=st.session_state.overlay_off,
        help="Display CPU/GPU usage statistics on video",
        key="overlay_toggle",
        disabled=st.session_state.disabled_params,
        on_change=disable_overlay
    )
    
    use_ema = st.toggle(
        "Use EMA Updates",
        value=True,
        help="Use Exponential Moving Average for feature updates (vs simple averaging)",
        on_change=disable_ema,
        key="ema_toggle",
        disabled=st.session_state.disabled_params,
    )
    use_gpu = st.toggle(
        "Use GPU",
        value=st.session_state.use_gpu,
        help="Disbables GPU usage even if available",
        key="gpu_toggle",
        on_change=disable_gpu,
        disabled=st.session_state.disabled_params,
    )
    
    st.markdown("---")
    
    # Process Button
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start", 
                              type="primary", 
                              use_container_width=True, 
                              on_click=params_disabled,
                              disabled=st.session_state.disabled_params
                              )
    with col2:
        stop_btn = st.button("⏹️ Stop", 
                             use_container_width=True,
                             on_click=params_disabled,
                             disabled=st.session_state.enable_stop_btn
                             )
        

def render_logs():
    """Always render the logs box with current content or idle message"""
    if st.session_state.logs:
        entries = ''.join(f'<div class="log-entry">{log}</div>'
                          for log in st.session_state.logs[-15:])  # chronological
    else:
        entries = '<div class="log-entry">Upload a video, set your desired parameters, and hit play. Experience ReID on your own video...</div>'
    
    log_html = f'''
    <div class="log-box" id="log-box">
        {entries}
    </div>
    <script>
        var logBox = document.getElementById('log-box');
        if (logBox) {{
            logBox.scrollTop = logBox.scrollHeight;
        }}
    </script>
    '''
    log_placeholder.markdown(log_html, unsafe_allow_html=True)

# --------------------------
# Main Layout
# --------------------------
# Video and Logs Section
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 🎬 Video Playback")
    video_placeholder = st.empty()
    
    # Progress indicators below video
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    
    with progress_col1:
        fps_metric = st.empty()
        fps_metric.metric("FPS", "0.0")
    
    with progress_col2:
        frames_metric = st.empty()
        frames_metric.metric("Frames", "0 / 0")
    
    with progress_col3:
        time_metric = st.empty()
        time_metric.metric("Elapsed", "00:00")
    
    with progress_col4:
        ids_metric = st.empty()
        ids_metric.metric("Tracked IDs", "0")
    
    # Progress bar
    progress_bar = st.empty()
    progress_bar.progress(0)

with col2:
    st.markdown("### 📊 Activity Logs")
    log_placeholder = st.empty()
    render_logs()

# --------------------------
# Hints Section
# --------------------------
st.markdown("---")
hints_container = st.container()
with hints_container:
    st.markdown("""
    <div class="hints-container">
        <div class="hint-title">💡 Usage Hints & Parameter Guide</div>
        <div class="hint-item"><strong>NMS Threshold:</strong> Controls detection confidence. Lower values = more detections but possibly more false positives.</div>
        <div class="hint-item"><strong>IoU Threshold:</strong> Determines when objects are considered occluded. Lower = more sensitive to occlusions.</div>
        <div class="hint-item"><strong>Cosine Similarity:</strong> Identity matching strictness. Higher = stricter matching, fewer ID switches but may miss re-identifications.</div>
        <div class="hint-item"><strong>EMA Alpha:</strong> Feature update rate. Higher = slower adaptation, more stable tracking. Lower = faster adaptation to appearance changes.</div>
        <div class="hint-item"><strong>Proximity Threshold:</strong> Maximum distance for considering re-acquisition of lost tracks.</div>
        <div class="hint-item"><strong>Use Case:</strong> Upload surveillance or tracking videos to test person re-identification performance in real-world scenarios.</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Helper Functions
# --------------------------

def format_time(seconds):
    """Format seconds to MM:SS"""
    return str(timedelta(seconds=int(seconds)))[2:7]

def check_reacquisition(pid, curr_bbox, frame_id, frame_shape, proximity_px):
    """Enhanced reacquisition detection with configurable proximity"""
    h, w = frame_shape[:2]
    
    x1, y1, x2, y2 = map(int, curr_bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    if pid in st.session_state.last_seen:
        gap = frame_id - st.session_state.last_seen[pid]
        
        # Short gap (occlusion recovery)
        if gap <= 3 and st.session_state.was_missing.get(pid, False):
            lx1, ly1, lx2, ly2 = map(int, st.session_state.last_bbox.get(pid, curr_bbox))
            lcx = (lx1 + lx2) / 2.0
            lcy = (ly1 + ly2) / 2.0
            dist = float(((cx - lcx)**2 + (cy - lcy)**2) ** 0.5)
            
            if dist < proximity_px:
                msg = f"[{time.strftime('%H:%M:%S')}] 🔄 ID {pid} recovered from occlusion"
                st.session_state.logs.append(msg)
                st.session_state.was_missing[pid] = False
        
        # Long gap (reacquisition)
        elif gap > 25 and st.session_state.was_missing.get(pid, False):
            msg = f"[{time.strftime('%H:%M:%S')}] 👤 ID {pid} re-acquired after {gap} frames"
            st.session_state.logs.append(msg)
            st.session_state.was_missing[pid] = False
    
    # Update tracking
    st.session_state.last_seen[pid] = frame_id
    st.session_state.last_bbox[pid] = (x1, y1, x2, y2)
    st.session_state.was_missing[pid] = False
###############################################################################################################
def run_inference(video_path):
    """Main inference loop with real-time parameter updates"""
    if st.session_state.use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize pipeline with current parameters
    st.session_state.pipeline = VideoReIDPipeline(
        det_model_path="src/data/models/yolov8n.pt",
        reid_weight_path="src/data/models/osnet.pth.tar-10",
        nms_th=st.session_state.parameters["nms_threshold"],
        device=device,
        cosine_th=st.session_state.parameters["cosine_threshold"],
        iou_th=st.session_state.parameters["iou_threshold"],
        ema_alpha=st.session_state.parameters["ema_alpha"],
        use_ema=not st.session_state.ema_off,
        use_overlay=st.session_state.overlay_off,
    )
    
    cap = cv2.VideoCapture(video_path)
    st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    st.session_state.start_time = time.time()
    st.session_state.is_processing = True
    frame_id = 0
    
    # Add processing start log
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Processing started - {st.session_state.total_frames} frames @ {fps_video:.1f} FPS")
    
    while cap.isOpened() and st.session_state.is_processing:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        st.session_state.current_frame = frame_id
        
        # Process frame
        vis_frame, assigned, valid_dets, occluded_indices = st.session_state.pipeline.process_frame(frame)
                
        # Track IDs and handle events
        current_ids = []
        for idx, (pid, _) in enumerate(assigned):
            current_ids.append(pid)
            
            # Check for occlusions
            if idx in occluded_indices:
                if pid not in st.session_state.was_missing:
                    msg = f"[{time.strftime('%H:%M:%S')}] ⚠️ ID {pid} occluded at frame {frame_id}"
                    st.session_state.logs.append(msg)
            
            # Check reacquisition
            x1, y1, x2, y2 = valid_dets[idx]['bbox']
            check_reacquisition(pid, (x1, y1, x2, y2), frame_id, frame.shape, st.session_state.parameters["proximity_threshold"])
        
        # Update ID streaks for new confirmations
        for pid in current_ids:
            if pid in st.session_state.confirmed_ids:
                continue
            st.session_state.id_streaks[pid] = st.session_state.id_streaks.get(pid, 0) + 1
            if st.session_state.id_streaks[pid] == 10:
                msg = f"[{time.strftime('%H:%M:%S')}] ✨ New person detected: ID {pid}"
                st.session_state.logs.append(msg)
                st.session_state.confirmed_ids.add(pid)
        
        # Mark missing IDs
        for pid in list(st.session_state.last_seen.keys()):
            if pid not in current_ids:
                st.session_state.was_missing[pid] = True
        
        # Update display
        video_placeholder.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
        
        # Update metrics
        elapsed = time.time() - st.session_state.start_time
        current_fps = frame_id / elapsed if elapsed > 0 else 0
        
        fps_metric.metric("FPS", f"{current_fps:.1f}")
        frames_metric.metric("Frames", f"{frame_id} / {st.session_state.total_frames}")
        time_metric.metric("Elapsed", format_time(elapsed))
        ids_metric.metric("Tracked IDs", len(st.session_state.confirmed_ids))
        
        # Update progress bar
        progress = frame_id / st.session_state.total_frames if st.session_state.total_frames > 0 else 0
        progress_bar.progress(progress)
        
        # Update logs
        log_html = f'''
            <div class="log-box" id="log-box">
                {''.join(f'<div class="log-entry">{log}</div>' for log in st.session_state.logs[-15:])}
            </div>
            <script>
                var logBox = document.getElementById('log-box');
                logBox.scrollTop = logBox.scrollHeight;
            </script>
        '''
        log_placeholder.markdown(log_html, unsafe_allow_html=True)
    
    cap.release()
    st.session_state.is_processing = False
    
    # Final log
    total_time = time.time() - st.session_state.start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Processing complete - {frame_id} frames in {format_time(total_time)} (Avg: {avg_fps:.1f} FPS)")

# --------------------------
# Handle Video Upload and Display
# --------------------------
if video_file is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
    tfile.write(video_file.read())
    video_path = tfile.name
    
    if not st.session_state.is_processing:
        # Display video preview
        video_placeholder.video(video_file)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Update initial metrics
        fps_metric.metric("FPS", f"{fps:.1f}")
        frames_metric.metric("Frames", f"0 / {total_frames}")
else:
    # Empty state
    video_placeholder.info("📹 Please upload a video to begin processing")
    log_html = '<div class="log-box">Upload a video, set your desired parameters, and hit play. Experience ReID on your own video...</div>'
    log_placeholder.markdown(log_html, unsafe_allow_html=True)

# --------------------------
# Handle Processing Controls
# --------------------------

if start_btn and video_file is not None:
    # Reset logs and state before processing
    st.session_state.logs = []
    st.session_state.id_streaks = {}
    st.session_state.confirmed_ids = set()
    st.session_state.last_seen = {}
    st.session_state.last_bbox = {}
    st.session_state.was_missing = {}

    # Check GPU availability and user preference
    device = "cuda" if torch.cuda.is_available() and st.session_state.use_gpu else "cpu"
    
    # Log device selection
    if device == "cuda":
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Using GPU for processing...")
    else:
        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Using CPU for processing...")
    
    # Add configuration log
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] 📂 Processing: {video_file.name}\n"
        f"Parameters: NMS={st.session_state.parameters['nms_threshold']:.2f}, "
        f"Device={device}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] #########   Parameters Used   #########"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] Detector NMS={st.session_state.parameters['nms_threshold']:.2f}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] Use GPU={st.session_state.use_gpu:}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] IoU Threshold={st.session_state.parameters['iou_threshold']:.2f}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] Use EMA={st.session_state.ema_off:.2f}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] EMA Alpha={st.session_state.parameters['ema_alpha']:.2f}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] Cosine Similarity Threshold={st.session_state.parameters['cosine_threshold']:.2f}"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] Proximity Threshold={st.session_state.parameters['proximity_threshold']:.2f} pixels"
    )
    st.session_state.logs.append(
        f"[{time.strftime('%H:%M:%S')}] ########################################"
    )
    
    run_inference(video_path)

if stop_btn and st.session_state.is_processing:
    st.session_state.is_processing = False
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] ⏹️ Processing stopped by user")

    video_placeholder.info("📹 Please upload a video to begin processing")
    log_html = '<div class="log-box">Session was stopped by the user...</div>'
    log_placeholder.markdown(log_html, unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("""
<div class="footer">
    TRIDENT AI Vision System v1.0 | Powered by YOLOv8 + OSNet | Built with Streamlit
</div>
""", unsafe_allow_html=True)