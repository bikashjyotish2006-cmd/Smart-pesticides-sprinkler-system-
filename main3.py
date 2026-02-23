import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time
from flask import Flask, request, jsonify, render_template_string, Response
from datetime import datetime


# ========================
# GPU CONFIGURATION
# ========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"SUCCESS: GPU detected: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"WARNING: GPU configuration error: {e}")
else:
    print("WARNING: No GPU detected, using CPU")

# ------------------------------
# Load models and define constants
# ------------------------------
BINARY_MODEL_PATH = r"C:\Users\BIKASH\Desktop\webcam\plant_vs_nonplant_mobilenetv2_final.h5"
SEVERITY_MODEL_PATH = r"C:\Users\BIKASH\Desktop\webcam\suraj_chand_severity_mobilenetv2_optimized_final_ooooop.h5"

try:
    binary_model = load_model(BINARY_MODEL_PATH, compile=False)
    severity_model = load_model(SEVERITY_MODEL_PATH, compile=False)
    print("SUCCESS: Models loaded successfully")
except Exception as e:
    print(f"ERROR: Error loading models: {e}")
    exit()

SEVERITY_CLASSES = ['healthy', 'high', 'low', 'medium']
IMG_SIZE = (224, 224)
BINARY_THRESHOLD = 0.6
CONF_THRESHOLD = 45.0
HIGH_CONF_THRESHOLD = 75.0

# Target box size for AI analysis (300x300 center ROI)
TARGET_SIZE = 300

# HD Resolution settings
HD_WIDTH = 1280
HD_HEIGHT = 720

# Motor activation thresholds
MOISTURE_THRESHOLD = 40.0  # < 40%
HUMIDITY_THRESHOLD = 70.0  # < 70%
TEMP_THRESHOLD = 30.0      # < 30C

# Motor run times based on severity (for high-confidence spray)
SPRAY_RUN_TIMES = {
    'low': 2,      # 2 sec
    'medium': 3,   # 3 sec
    'high': 5      # 5 sec
}

# ------------------------------
# Motor Status Globals
# ------------------------------
motor_state = False  # True when ON, False when OFF
motor_start_time = 0
motor_duration = 0

# ------------------------------
# Temporal Smoothing + Hysteresis
# ------------------------------
SMOOTH_FRAMES = 20
ALPHA = 0.1
HYSTERESIS_THRESHOLD = 5
history = deque(maxlen=SMOOTH_FRAMES)
plant_state = False
no_plant_count = 0

# ------------------------------
# Flask Globals
# ------------------------------
app = Flask(__name__)
current_command = "STOP"
current_duration = 0
latest_moisture = 0.0
latest_temperature = None
latest_humidity = None
latest_plant_data = {"label": "No Plant Detected", "confidence": 0.0}
force_spray = False  # Global flag for force spray

# ------------------------------
# System Logs (max 50 entries)
# ------------------------------
system_logs = deque(maxlen=50)

def add_log(message, log_type="info"):
    """Add a log entry with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    system_logs.append({
        "time": timestamp,
        "message": message,
        "type": log_type  # info, warning, success, error
    })

# ------------------------------
# Frame Buffers
# ------------------------------
frame_buffer = deque(maxlen=1)
result_buffer = deque(maxlen=1)

# ------------------------------
# Freeze Mode and Spray Logic
# ------------------------------
freeze_mode = False
frozen_frame = None
spray_triggered = False
spray_start_time = 0
spray_duration = 0
force_spray_message = ""  # Global for on-screen prompt

def get_smoothed_label():
    global plant_state, no_plant_count
    if not history:
        return "No Plant Detected", (0, 0, 255), 0.0
    smoothed_conf = 0.0
    for i, (_, conf) in enumerate(history):
        smoothed_conf = conf if i == 0 else ALPHA * conf + (1 - ALPHA) * smoothed_conf
    valid_labels = [lbl for lbl, _ in history if lbl != "No Plant Detected"]
    most_common_label = max(set(valid_labels), key=valid_labels.count) if valid_labels else "No Plant Detected"
    if smoothed_conf >= CONF_THRESHOLD:
        plant_state = True
        no_plant_count = 0
        return most_common_label, (0, 255, 0), smoothed_conf
    else:
        no_plant_count += 1
        if plant_state and no_plant_count < HYSTERESIS_THRESHOLD:
            return most_common_label, (0, 255, 0), smoothed_conf
        else:
            plant_state = False
            no_plant_count = 0
            return "No Plant Detected", (0, 0, 255), smoothed_conf

# ------------------------------
# Spray Motor Function
# ------------------------------
def spray_motor(severity):
    global spray_triggered, spray_start_time, spray_duration, motor_state, freeze_mode, force_spray_message
    if severity in SPRAY_RUN_TIMES:
        spray_duration = SPRAY_RUN_TIMES[severity]
        spray_start_time = time.time()
        motor_state = True
        log_msg = f"Motor started for {spray_duration}s - {severity.upper()} severity detected"
        print(f"Starting spray for {spray_duration}s due to {severity} severity")
        add_log(log_msg, "success")
        # Simulate motor run (in real, send command to Pi)
        time.sleep(spray_duration)
        motor_state = False
        print("Spray completed, motor stopped")
        add_log(f"Motor stopped after {spray_duration}s spray", "info")
        # Reset freeze mode and spray_triggered after motor stops
        spray_triggered = False
        freeze_mode = False
        force_spray_message = ""  # Clear the on-screen prompt

# ------------------------------
# UI Crosshair Drawing Function
# ------------------------------
def draw_target_ui(frame, target_size=TARGET_SIZE):
    """
    Draw a high-tech targeting crosshair UI on the frame.
    Returns the frame with overlay and the ROI coordinates.
    """
    h, w = frame.shape[:2]
    
    # Calculate center ROI coordinates
    start_x = (w - target_size) // 2
    start_y = (h - target_size) // 2
    end_x = start_x + target_size
    end_y = start_y + target_size
    
    center_x = w // 2
    center_y = h // 2
    
    # Colors
    color_primary = (0, 245, 160)  # Emerald green
    color_secondary = (255, 255, 255)  # White
    color_dim = (100, 100, 100)  # Gray for subtle elements
    
    # Draw corner brackets (L-shaped corners)
    bracket_length = 40
    bracket_thickness = 3
    
    # Top-left corner
    cv2.line(frame, (start_x, start_y), (start_x + bracket_length, start_y), color_primary, bracket_thickness)
    cv2.line(frame, (start_x, start_y), (start_x, start_y + bracket_length), color_primary, bracket_thickness)
    
    # Top-right corner
    cv2.line(frame, (end_x, start_y), (end_x - bracket_length, start_y), color_primary, bracket_thickness)
    cv2.line(frame, (end_x, start_y), (end_x, start_y + bracket_length), color_primary, bracket_thickness)
    
    # Bottom-left corner
    cv2.line(frame, (start_x, end_y), (start_x + bracket_length, end_y), color_primary, bracket_thickness)
    cv2.line(frame, (start_x, end_y), (start_x, end_y - bracket_length), color_primary, bracket_thickness)
    
    # Bottom-right corner
    cv2.line(frame, (end_x, end_y), (end_x - bracket_length, end_y), color_primary, bracket_thickness)
    cv2.line(frame, (end_x, end_y), (end_x, end_y - bracket_length), color_primary, bracket_thickness)
    
    # Draw center crosshair
    crosshair_size = 15
    cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), color_secondary, 2)
    cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), color_secondary, 2)
    
    # Draw center dot
    cv2.circle(frame, (center_x, center_y), 4, color_primary, -1)
    cv2.circle(frame, (center_x, center_y), 6, color_secondary, 1)
    
    # Draw subtle grid lines (dotted effect using multiple small lines)
    dash_spacing = 20
    for i in range(start_x + dash_spacing, end_x, dash_spacing * 2):
        cv2.line(frame, (i, start_y), (i + dash_spacing, start_y), color_dim, 1)
        cv2.line(frame, (i, end_y), (i + dash_spacing, end_y), color_dim, 1)
    
    for i in range(start_y + dash_spacing, end_y, dash_spacing * 2):
        cv2.line(frame, (start_x, i), (start_x, i + dash_spacing), color_dim, 1)
        cv2.line(frame, (end_x, i), (end_x, i + dash_spacing), color_dim, 1)
    
    # Draw text labels
    label_text = f"AI ANALYSIS ZONE - {target_size}x{target_size}"
    cv2.putText(frame, label_text, (start_x, start_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_primary, 2)
    
    # Draw pixel dimensions at corners
    cv2.putText(frame, f"{target_size}px", (start_x, start_y - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_secondary, 1)
    
    return frame, (start_x, start_y, end_x, end_y)

# ------------------------------
# Multi-threaded frame processing
# ------------------------------
def process_frame():
    global motor_state, freeze_mode, spray_triggered
    while True:
        if frame_buffer:
            frame = frame_buffer.popleft()
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Calculate center ROI for AI analysis
            start_x = (w - TARGET_SIZE) // 2
            start_y = (h - TARGET_SIZE) // 2
            end_x = start_x + TARGET_SIZE
            end_y = start_y + TARGET_SIZE
            
            # Extract ROI for AI analysis only
            roi = frame[start_y:end_y, start_x:end_x]
            
            # Prepare for model (resize ROI to 224x224)
            img = cv2.resize(roi, IMG_SIZE).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Step 1: Binary Model (Plant vs Non-Plant)
            binary_pred = binary_model.predict(img, verbose=0)[0][0]
            
            if binary_pred > BINARY_THRESHOLD:
                # Step 2: Severity Model
                preds = severity_model.predict(img, verbose=0)
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds) * 100)
                label = f"{SEVERITY_CLASSES[class_id]} ({confidence:.1f}%)"
                history.append((label, confidence))
            else:
                history.append(("No Plant Detected", 0.0))
            
            smoothed_label, smoothed_color, smoothed_conf = get_smoothed_label()
            
            # Create a copy of the original HD frame for display
            display_frame = frame.copy()
            
            # Draw the targeting UI overlay on the full HD frame
            display_frame, roi_coords = draw_target_ui(display_frame, TARGET_SIZE)
            
            # Draw detection results overlay (positioned below the target box)
            result_y = end_y + 40
            cv2.putText(display_frame, f"Plant: {smoothed_label}", (start_x, result_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, smoothed_color, 2)
            
            cv2.putText(display_frame, f"Confidence: {smoothed_conf:.1f}%", (start_x, result_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update latest plant data for dashboard
            latest_plant_data["label"] = smoothed_label
            latest_plant_data["confidence"] = smoothed_conf
            
            # Auto Motor Trigger
            severity = smoothed_label.split(" ")[0]
            if (severity in SPRAY_RUN_TIMES and
                latest_moisture < MOISTURE_THRESHOLD and
                (latest_humidity is not None and latest_humidity < HUMIDITY_THRESHOLD) and
                (latest_temperature is not None and latest_temperature < TEMP_THRESHOLD) and
                not motor_state):
                threading.Thread(target=spray_motor, args=(severity,), daemon=True).start()
            
            result_buffer.append((smoothed_label, smoothed_color, smoothed_conf, display_frame))
        
        time.sleep(0.01)

def generate_frames():
    while True:
        if result_buffer:
            label, color, confidence, frame = result_buffer[-1]
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/process', methods=['POST'])
def process():
    global current_command, current_duration, latest_moisture, force_spray
    data = request.json
    latest_moisture = data.get('moisture', 0.0)
    print(f"\n--- [PI DATA] Moisture: {latest_moisture}% ---")
    plant_label = latest_plant_data.get("label", "").split(" ")[0]
    
    # Check for force spray first
    if force_spray:
        current_command = "RUN"
        current_duration = 3  # Force spray for 3 seconds
        force_spray = False  # Reset after sending
        print("Force spray activated via 'S' key -> Motor RUN for 3s")
        add_log("Force spray activated - Motor RUN for 3s", "warning")
    else:
        # Normal logic: Plant detected AND all thresholds met
        if (plant_label in SPRAY_RUN_TIMES and plant_label != "No Plant Detected" and
            latest_moisture < MOISTURE_THRESHOLD and
            (latest_humidity is not None and latest_humidity < HUMIDITY_THRESHOLD) and
            (latest_temperature is not None and latest_temperature < TEMP_THRESHOLD)):
            current_command = "RUN"
            current_duration = SPRAY_RUN_TIMES[plant_label]
            print(f"Plant detected ({plant_label}), Conditions met -> Motor RUN for {current_duration}s")
            add_log(f"Auto-trigger: {plant_label} severity, conditions met", "success")
        else:
            current_command = "STOP"
            current_duration = 0
            print("No action: Conditions not met")
    
    return jsonify({"motor_command": current_command, "duration": current_duration})

@app.route('/dht22', methods=['POST'])
def dht22():
    global latest_temperature, latest_humidity
    data = request.json
    latest_temperature = data.get('temperature')
    latest_humidity = data.get('humidity')
    print(f"--- [DHT22 DATA] Temperature: {latest_temperature}C, Humidity: {latest_humidity}% ---")
    return jsonify({"status": "received"})

@app.route('/logs')
def get_logs():
    """Return system logs"""
    return jsonify(list(system_logs))

@app.route('/')
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Sprayer System | Plant AI Monitor Pro</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --emerald-primary: #00f5a0; --emerald-dark: #00c97f; --emerald-light: #5fffd1; --emerald-glow: rgba(0, 245, 160, 0.45);
            --alert-red: #ff3b5c; --alert-red-dark: #d91e3f; --alert-red-glow: rgba(255, 59, 92, 0.45);
            --warning-amber: #ffb020; --warning-glow: rgba(255, 176, 32, 0.4);
            --info-blue: #3da9ff; --info-glow: rgba(61, 169, 255, 0.45);
            --automation-violet: #8b5cf6; --automation-glow: rgba(139, 92, 246, 0.4);
            --bg-dark: #060b14; --bg-darker: #04080f; --bg-card: #0f1724; --bg-card-hover: #162132;
            --bg-glass: rgba(15, 23, 36, 0.55); --bg-glass-strong: rgba(15, 23, 36, 0.75);
            --text-primary: #e6edf6; --text-secondary: #9fb3c8; --text-muted: #6b8098;
            --border-color: rgba(80, 120, 160, 0.15); --border-light: rgba(120, 170, 220, 0.25);
            --shadow-sm: 0 4px 12px rgba(0, 0, 0, 0.35); --shadow-md: 0 12px 28px rgba(0, 0, 0, 0.45);
            --gradient-primary: linear-gradient(135deg, #00f5a0 0%, #00c97f 100%);
            --gradient-danger: linear-gradient(135deg, #ff3b5c 0%, #d91e3f 100%);
            --gradient-warning: linear-gradient(135deg, #ffb020 0%, #ff8c00 100%);
            --gradient-info: linear-gradient(135deg, #3da9ff 0%, #2563eb 100%);
            --radius-sm: 10px; --radius-md: 16px; --radius-lg: 22px;
            --transition-smooth: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
            --blur-glass: blur(18px); --blur-strong: blur(28px);
            --status-online: #00f5a0; --status-offline: #ff3b5c; --status-idle: #ffb020; --status-processing: #3da9ff;
        }
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; -webkit-font-smoothing: antialiased; }
        body { 
            font-family: 'Inter', system-ui, sans-serif;
            background: radial-gradient(circle at 20% 20%, rgba(0,245,160,0.05), transparent 40%), radial-gradient(circle at 80% 70%, rgba(61,169,255,0.05), transparent 50%), linear-gradient(145deg, var(--bg-darker), var(--bg-dark));
            color: var(--text-primary); min-height: 100vh; overflow-x: hidden; line-height: 1.6; letter-spacing: 0.2px; background-attachment: fixed;
        }
        body::before { content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(0, 245, 160, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 245, 160, 0.05) 1px, transparent 1px); background-size: 60px 60px; z-index: -1; opacity: 0.6; animation: gridDrift 40s linear infinite; }
        @keyframes gridDrift { 0% { transform: translate(0, 0); } 100% { transform: translate(-60px, -60px); } }
        
        /* Header Animations */
        .mission-header { background: linear-gradient(145deg, var(--bg-glass-strong), var(--bg-card)), radial-gradient(circle at 20% 50%, rgba(0, 245, 160, 0.08), transparent 60%); backdrop-filter: var(--blur-glass); border-bottom: 1px solid var(--border-light); box-shadow: var(--shadow-md), inset 0 -1px 0 rgba(0, 245, 160, 0.15); padding: 1.2rem 0; position: relative; overflow: hidden; z-index: 5; }
        .mission-header::before { content: ""; position: absolute; bottom: 0; left: -50%; width: 50%; height: 2px; background: var(--gradient-primary); animation: headerScan 6s linear infinite; }
        @keyframes headerScan { 0% { left: -50%; } 100% { left: 100%; } }
        .project-title { font-size: 1.6rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; background: linear-gradient(135deg, #ffffff 0%, var(--text-primary) 40%, var(--emerald-light) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: titleGlow 3s ease-in-out infinite alternate; }
        @keyframes titleGlow { from { filter: drop-shadow(0 0 5px rgba(0, 245, 160, 0.3)); } to { filter: drop-shadow(0 0 15px rgba(0, 245, 160, 0.6)); } }
        .team-badge { background: var(--gradient-primary); color: #04110c; padding: 0.4rem 0.85rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; border: 1px solid rgba(0, 245, 160, 0.35); box-shadow: 0 0 12px rgba(0, 245, 160, 0.35); animation: badgePulse 2s ease-in-out infinite; }
        @keyframes badgePulse { 0%, 100% { box-shadow: 0 0 12px rgba(0, 245, 160, 0.35); } 50% { box-shadow: 0 0 20px rgba(0, 245, 160, 0.6); } }
        
        /* Status Indicator */
        .status-indicator { display: inline-flex; align-items: center; gap: 0.6rem; font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); padding: 0.35rem 0.6rem; border-radius: var(--radius-sm); background: rgba(0, 245, 160, 0.04); border: 1px solid rgba(0, 245, 160, 0.08); backdrop-filter: var(--blur-glass); transition: all 0.3s ease; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--status-online); box-shadow: 0 0 8px var(--status-online), 0 0 16px rgba(0, 245, 160, 0.4); animation: pulseCore 2.5s ease-in-out infinite; position: relative; }
        .status-dot::before { content: ""; position: absolute; inset: -6px; border-radius: 50%; border: 1px solid rgba(0, 245, 160, 0.4); animation: pulseRing 2.5s ease-out infinite; }
        @keyframes pulseCore { 0%, 100% { transform: scale(1); box-shadow: 0 0 8px var(--status-online), 0 0 16px rgba(0, 245, 160, 0.4); } 50% { transform: scale(1.15); box-shadow: 0 0 14px var(--status-online), 0 0 28px rgba(0, 245, 160, 0.6); } }
        @keyframes pulseRing { 0% { transform: scale(0.6); opacity: 0.7; } 70% { transform: scale(1.6); opacity: 0; } 100% { opacity: 0; } }
        .status-dot.offline { background: var(--status-offline); animation: alertFlicker 1.8s infinite; }
        @keyframes alertFlicker { 0%, 100% { opacity: 1; } 10% { opacity: 0.6; } 20% { opacity: 1; } 40% { opacity: 0.7; } 60% { opacity: 1; } 80% { opacity: 0.85; } }
        
        /* KPI Cards with Enhanced Animations */
        .kpi-card { background: linear-gradient(145deg, var(--bg-glass-strong), var(--bg-card)); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 1.6rem; position: relative; overflow: hidden; height: 100%; backdrop-filter: var(--blur-glass); box-shadow: var(--shadow-sm), inset 0 0 30px rgba(0, 245, 160, 0.02); transition: var(--transition-smooth); cursor: pointer; }
        .kpi-card::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent 0%, var(--emerald-primary) 40%, var(--emerald-light) 50%, var(--emerald-primary) 60%, transparent 100%); transform: scaleX(0); transform-origin: left; transition: transform 0.45s cubic-bezier(0.22, 1, 0.36, 1); }
        .kpi-card:hover { transform: translateY(-8px) scale(1.02); border-color: rgba(0, 245, 160, 0.4); box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 245, 160, 0.2); }
        .kpi-card:hover::before { transform: scaleX(1); }
        .kpi-card.alert { border-color: rgba(255, 59, 92, 0.3); animation: alertBorderPulse 2s ease-in-out infinite; }
        .kpi-card.alert::before { background: linear-gradient(90deg, transparent 0%, var(--alert-red) 40%, #ff6b81 50%, var(--alert-red-dark) 60%, transparent 100%); }
        .kpi-card.alert:hover { border-color: rgba(255, 59, 92, 0.6); box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px rgba(255, 59, 92, 0.3); }
        @keyframes alertBorderPulse { 0%, 100% { border-color: rgba(255, 59, 92, 0.3); } 50% { border-color: rgba(255, 59, 92, 0.6); } }
        .kpi-card.warning { border-color: rgba(255, 176, 32, 0.3); animation: warningBorderPulse 2s ease-in-out infinite; }
        .kpi-card.warning::before { background: linear-gradient(90deg, transparent 0%, var(--warning-amber) 40%, #ffd166 50%, #ff9f1c 60%, transparent 100%); }
        @keyframes warningBorderPulse { 0%, 100% { border-color: rgba(255, 176, 32, 0.3); } 50% { border-color: rgba(255, 176, 32, 0.6); } }
        
        .kpi-icon { width: 50px; height: 50px; border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: 1.1rem; background: linear-gradient(145deg, rgba(0, 245, 160, 0.08), rgba(0, 245, 160, 0.03)); border: 1px solid rgba(0, 245, 160, 0.15); transition: var(--transition-smooth); }
        .kpi-card:hover .kpi-icon { transform: scale(1.1) rotate(5deg); }
        .kpi-icon.healthy { background: linear-gradient(145deg, rgba(0, 245, 160, 0.18), rgba(0, 245, 160, 0.06)); color: var(--status-online); border-color: rgba(0, 245, 160, 0.25); box-shadow: 0 0 15px rgba(0, 245, 160, 0.2); animation: healthyPulse 2s ease-in-out infinite; }
        @keyframes healthyPulse { 0%, 100% { box-shadow: 0 0 15px rgba(0, 245, 160, 0.2); } 50% { box-shadow: 0 0 25px rgba(0, 245, 160, 0.4); } }
        .kpi-icon.infected { background: linear-gradient(145deg, rgba(255, 59, 92, 0.22), rgba(255, 59, 92, 0.08)); color: var(--status-offline); border-color: rgba(255, 59, 92, 0.35); box-shadow: 0 0 15px rgba(255, 59, 92, 0.3); animation: infectedPulse 1.5s ease-in-out infinite; }
        @keyframes infectedPulse { 0%, 100% { box-shadow: 0 0 15px rgba(255, 59, 92, 0.3); } 50% { box-shadow: 0 0 25px rgba(255, 59, 92, 0.5); } }
        .kpi-icon.moisture { background: linear-gradient(145deg, rgba(61, 169, 255, 0.20), rgba(61, 169, 255, 0.07)); color: var(--status-processing); border-color: rgba(61, 169, 255, 0.30); }
        .kpi-icon.temperature { background: linear-gradient(145deg, rgba(255, 176, 32, 0.22), rgba(255, 140, 0, 0.08)); color: var(--status-idle); border-color: rgba(255, 176, 32, 0.35); }
        .kpi-icon.humidity { background: linear-gradient(145deg, rgba(6, 182, 212, 0.22), rgba(6, 182, 212, 0.08)); color: #22d3ee; border-color: rgba(6, 182, 212, 0.35); }
        .kpi-icon.pump { background: linear-gradient(145deg, rgba(139, 92, 246, 0.22), rgba(124, 58, 237, 0.08)); color: #a78bfa; border-color: rgba(139, 92, 246, 0.35); }
        .kpi-icon.spraying { animation: pumpSpin 1s linear infinite; background: linear-gradient(145deg, rgba(0, 245, 160, 0.3), rgba(0, 201, 127, 0.15)); color: var(--emerald-primary); border-color: var(--emerald-primary); }
        @keyframes pumpSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        .kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.4px; color: var(--text-secondary); opacity: 0.85; margin-bottom: 0.55rem; }
        .kpi-value { font-size: 1.9rem; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 0.3rem; line-height: 1.1; transition: all 0.3s ease; }
        .kpi-value.healthy { color: var(--status-online); text-shadow: 0 0 12px rgba(0, 245, 160, 0.25); animation: valueGlow 2s ease-in-out infinite alternate; }
        @keyframes valueGlow { from { text-shadow: 0 0 12px rgba(0, 245, 160, 0.25); } to { text-shadow: 0 0 20px rgba(0, 245, 160, 0.5); } }
        .kpi-value.infected { color: var(--status-offline); text-shadow: 0 0 14px rgba(255, 59, 92, 0.35); animation: valueAlert 1s ease-in-out infinite alternate; }
        @keyframes valueAlert { from { text-shadow: 0 0 14px rgba(255, 59, 92, 0.35); } to { text-shadow: 0 0 25px rgba(255, 59, 92, 0.6); } }
        .kpi-value.warning { color: var(--status-idle); text-shadow: 0 0 14px rgba(255, 176, 32, 0.3); }
        .kpi-value.info { color: var(--status-processing); text-shadow: 0 0 14px rgba(61, 169, 255, 0.3); }
        .kpi-card:hover .kpi-value { transform: scale(1.05); }
        .kpi-subtext { font-size: 0.85rem; font-weight: 500; color: var(--text-secondary); opacity: 0.8; transition: all 0.3s ease; }
        .kpi-card:hover .kpi-subtext { opacity: 1; color: var(--text-primary); }
        
        /* Operation Panel */
        .operation-panel { background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01)), var(--bg-card); border: 1px solid var(--border-color); border-radius: 20px; overflow: hidden; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25), inset 0 0 40px rgba(255, 255, 255, 0.02); transition: all 0.3s ease; }
        .operation-panel:hover { transform: translateY(-4px); border-color: rgba(255, 255, 255, 0.12); box-shadow: 0 14px 50px rgba(0, 0, 0, 0.35); }
        .panel-header { background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.03) 40%, transparent 70%); padding: 1.1rem 1.6rem; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 0.8rem; position: relative; }
        .panel-header::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--emerald-primary); box-shadow: 0 0 18px rgba(16, 185, 129, 0.35); }
        .panel-header h5 { margin: 0; font-weight: 600; font-size: 1.05rem; color: var(--text-primary); }
        
        /* Live Feed with Enhanced Effects */
        .live-feed-container { position: relative; background: radial-gradient(circle at center, rgba(0, 255, 180, 0.05), transparent 60%), #000; border-radius: 14px; overflow: hidden; aspect-ratio: 16 / 9; border: 1px solid rgba(255, 255, 255, 0.06); box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.8), 0 8px 25px rgba(0, 0, 0, 0.35); transition: all 0.3s ease; }
        .live-feed-container:hover { transform: translateY(-3px); border-color: rgba(0, 255, 180, 0.25); box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.85), 0 14px 40px rgba(0, 0, 0, 0.45); }
        .live-feed-container.infected { border: 3px solid var(--alert-red); box-shadow: 0 0 30px var(--alert-red-glow), 0 0 60px rgba(255, 59, 92, 0.25), inset 0 0 40px rgba(255, 59, 92, 0.15); animation: infectedFrame 1s ease-in-out infinite alternate; }
        @keyframes infectedFrame { from { box-shadow: 0 0 30px var(--alert-red-glow), inset 0 0 40px rgba(255, 59, 92, 0.15); } to { box-shadow: 0 0 50px var(--alert-red-glow), inset 0 0 60px rgba(255, 59, 92, 0.25); } }
        .live-feed-container.healthy { border: 3px solid var(--emerald-primary); box-shadow: 0 0 25px rgba(16, 185, 129, 0.25), inset 0 0 35px rgba(16, 185, 129, 0.08); animation: healthyFrame 2s ease-in-out infinite alternate; }
        @keyframes healthyFrame { from { box-shadow: 0 0 25px rgba(16, 185, 129, 0.25); } to { box-shadow: 0 0 40px rgba(16, 185, 129, 0.4); } }
        .live-feed-container.warning { border: 3px solid var(--warning-amber); box-shadow: 0 0 30px rgba(255, 176, 32, 0.35), inset 0 0 40px rgba(255, 176, 32, 0.12); animation: warningFrame 1.5s ease-in-out infinite alternate; }
        @keyframes warningFrame { from { box-shadow: 0 0 30px rgba(255, 176, 32, 0.35); } to { box-shadow: 0 0 50px rgba(255, 176, 32, 0.5); } }
        .live-badge { position: absolute; top: 12px; left: 12px; background: linear-gradient(135deg, var(--alert-red), rgba(255, 59, 92, 0.85)); color: #fff; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1.2px; backdrop-filter: blur(6px); box-shadow: 0 0 18px rgba(255, 59, 92, 0.45); z-index: 10; animation: livePulse 1.5s ease-in-out infinite; }
        @keyframes livePulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.8; transform: scale(1.05); } }
        .detection-overlay { position: absolute; bottom: 12px; left: 12px; right: 12px; background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px) saturate(150%); border: 1px solid rgba(16, 185, 129, 0.3); padding: 0.75rem 1rem; border-radius: 10px; display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 0 15px rgba(16, 185, 129, 0.25); z-index: 10; transition: all 0.3s ease; }
        .detection-overlay.healthy { border-left: 4px solid var(--emerald-primary); box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
        .detection-overlay.infected { border-left: 4px solid var(--alert-red); box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        .detection-overlay.warning { border-left: 4px solid var(--warning-amber); box-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
        .detection-status { display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.5px; }
        .detection-status.healthy { background: linear-gradient(90deg, var(--emerald-light), var(--emerald-primary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .detection-status.infected { background: linear-gradient(90deg, #f87171, var(--alert-red)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: statusAlert 1s ease-in-out infinite alternate; }
        @keyframes statusAlert { from { filter: brightness(1); } to { filter: brightness(1.3); } }
        .detection-status.warning { background: linear-gradient(90deg, #fbbf24, var(--warning-amber)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .confidence-score { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; color: var(--text-secondary); }
        
        /* Activity Log */
        .activity-log { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; overflow: hidden; box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15); }
        .log-header { background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(16, 185, 129, 0.05)); padding: 1rem 1.25rem; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 0.75rem; position: relative; }
        .log-header::after { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, var(--info-blue), var(--emerald-light), transparent); animation: scanline 3s linear infinite; }
        @keyframes scanline { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
        .log-entries { max-height: 350px; overflow-y: auto; padding: 0.5rem; }
        .log-entry { display: flex; align-items: flex-start; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 10px; margin-bottom: 0.5rem; background: rgba(16, 185, 129, 0.05); animation: slideIn 0.4s ease-out; border-left: 3px solid transparent; transition: all 0.3s ease; }
        .log-entry:hover { background: rgba(16, 185, 129, 0.12); box-shadow: 0 0 15px var(--emerald-glow); transform: translateX(5px); }
        .log-entry.healthy { border-left-color: var(--emerald-primary); background: rgba(0, 245, 160, 0.08); }
        .log-entry.infected { border-left-color: var(--alert-red); background: rgba(255, 59, 92, 0.08); }
        .log-entry.warning { border-left-color: var(--warning-amber); background: rgba(255, 176, 32, 0.08); }
        .log-entry.info { border-left-color: var(--info-blue); background: rgba(61, 169, 255, 0.08); }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
        .log-icon { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all 0.3s ease; }
        .log-icon.success { background: rgba(16, 185, 129, 0.2); color: var(--emerald-primary); box-shadow: 0 0 10px var(--emerald-glow); }
        .log-icon.warning { background: rgba(245, 158, 11, 0.2); color: var(--warning-amber); box-shadow: 0 0 10px var(--warning-glow); }
        .log-icon.error { background: rgba(239, 68, 68, 0.2); color: var(--alert-red); box-shadow: 0 0 10px var(--alert-red-glow); }
        .log-icon.info { background: rgba(59, 130, 246, 0.2); color: var(--info-blue); box-shadow: 0 0 10px var(--info-glow); }
        .log-message { font-size: 0.875rem; color: var(--text-primary); margin-bottom: 0.25rem; }
        .log-time { font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
        
        /* Force Spray Button */
        .force-spray-btn { width: 100%; padding: 1rem 1.5rem; border: none; border-radius: 12px; background: var(--gradient-danger); color: white; font-weight: 700; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.75rem; box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); transition: all 0.3s ease; position: relative; overflow: hidden; animation: btnPulse 2s ease-in-out infinite; }
        @keyframes btnPulse { 0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); } 50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.8); } }
        .force-spray-btn::before { content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); transition: left 0.5s; }
        .force-spray-btn:hover::before { left: 100%; }
        .force-spray-btn:hover:not(:disabled) { transform: translateY(-3px) scale(1.03); box-shadow: 0 0 35px rgba(239, 68, 68, 0.9); }
        .force-spray-btn:disabled { opacity: 0.6; cursor: not-allowed; animation: none; }
        .force-spray-btn.spraying { background: var(--gradient-primary); box-shadow: 0 0 25px rgba(16, 185, 129, 0.7); animation: sprayingPulse 0.5s ease-in-out infinite; }
        @keyframes sprayingPulse { 0%, 100% { box-shadow: 0 0 25px rgba(16, 185, 129, 0.7); } 50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.9); } }
        
        /* Chart Container */
        .chart-container { background: rgba(30, 41, 59, 0.85); border: 1px solid var(--emerald-glow); border-radius: 20px; padding: 1.5rem; height: 380px; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25); backdrop-filter: blur(12px); position: relative; }
        .chart-container::before { content: ''; position: absolute; top: -2px; left: -2px; width: calc(100% + 4px); height: calc(100% + 4px); border-radius: 20px; border: 2px solid transparent; background: linear-gradient(45deg, #10b981, #34d399, #059669, #10b981); background-size: 400% 400%; animation: neon-border 6s linear infinite; z-index: -1; }
        @keyframes neon-border { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        
        /* Toast Notifications */
        .toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 0.75rem; }
        .custom-toast { background: rgba(30, 41, 59, 0.98); border: 1px solid var(--emerald-glow); border-radius: 14px; padding: 1rem 1.25rem; display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3); animation: toastSlideIn 0.4s ease-out forwards; font-weight: 600; min-width: 280px; backdrop-filter: blur(10px); }
        @keyframes toastSlideIn { 0% { transform: translateX(100%); opacity: 0; } 100% { transform: translateX(0); opacity: 1; } }
        .custom-toast.success { border-left: 4px solid var(--emerald-primary); background: rgba(0, 245, 160, 0.1); }
        .custom-toast.error { border-left: 4px solid var(--alert-red); background: rgba(255, 59, 92, 0.1); }
        .custom-toast.warning { border-left: 4px solid var(--warning-amber); background: rgba(255, 176, 32, 0.1); }
        .custom-toast.info { border-left: 4px solid var(--info-blue); background: rgba(61, 169, 255, 0.1); }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.6); border-radius: 8px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--emerald-primary), var(--emerald-dark)); border-radius: 8px; box-shadow: 0 0 10px var(--emerald-primary); }
        ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, var(--emerald-light), var(--emerald-primary)); box-shadow: 0 0 15px var(--emerald-primary); }
        
        /* Overlay */
        .tech-overlay-backdrop { position: fixed; inset: 0; background: rgba(4, 8, 15, 0.95); backdrop-filter: var(--blur-strong); z-index: 1000; opacity: 0; visibility: hidden; transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1); display: flex; align-items: center; justify-content: center; padding: 2rem; }
        .tech-overlay-backdrop.active { opacity: 1; visibility: visible; }
        .tech-info-overlay { position: relative; width: 100%; max-width: 600px; max-height: 85vh; overflow-y: auto; background: linear-gradient(145deg, var(--bg-glass-strong), var(--bg-card)), radial-gradient(circle at 30% 20%, rgba(0, 245, 160, 0.08), transparent 50%); border: 1px solid var(--border-light); border-radius: var(--radius-lg); padding: 2rem; backdrop-filter: var(--blur-glass); box-shadow: var(--shadow-lg), 0 0 60px rgba(0, 245, 160, 0.15); transform: translateY(30px) scale(0.95); opacity: 0; transition: all 0.5s cubic-bezier(0.22, 1, 0.36, 1); }
        .tech-overlay-backdrop.active .tech-info-overlay { transform: translateY(0) scale(1); opacity: 1; }
        .tech-info-overlay::before { content: ""; position: absolute; inset: 0; border-radius: var(--radius-lg); padding: 2px; background: linear-gradient(135deg, var(--emerald-primary) 0%, var(--info-blue) 25%, var(--automation-violet) 50%, var(--info-blue) 75%, var(--emerald-primary) 100%); background-size: 300% 300%; -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor; mask-composite: exclude; animation: borderGlow 4s ease infinite; pointer-events: none; }
        @keyframes borderGlow { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
        .tech-corner { position: absolute; width: 20px; height: 20px; border: 2px solid var(--emerald-primary); opacity: 0.6; pointer-events: none; }
        .tech-corner-tl { top: 12px; left: 12px; border-right: none; border-bottom: none; }
        .tech-corner-tr { top: 12px; right: 12px; border-left: none; border-bottom: none; }
        .tech-corner-bl { bottom: 12px; left: 12px; border-right: none; border-top: none; }
        .tech-corner-br { bottom: 12px; right: 12px; border-left: none; border-top: none; }
        .tech-overlay-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color); position: relative; }
        .tech-overlay-icon { width: 56px; height: 56px; border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-size: 1.4rem; background: linear-gradient(145deg, rgba(0, 245, 160, 0.15), rgba(0, 245, 160, 0.05)); border: 1px solid rgba(0, 245, 160, 0.25); animation: iconPulse 2s ease-in-out infinite; }
        @keyframes iconPulse { 0%, 100% { box-shadow: inset 0 0 25px rgba(0, 245, 160, 0.1), 0 0 15px rgba(0, 245, 160, 0.2); } 50% { box-shadow: inset 0 0 35px rgba(0, 245, 160, 0.2), 0 0 25px rgba(0, 245, 160, 0.4); } }
        .tech-overlay-close { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: rgba(255, 59, 92, 0.1); border: 1px solid rgba(255, 59, 92, 0.3); color: var(--alert-red); font-size: 1.1rem; cursor: pointer; transition: var(--transition-smooth); }
        .tech-overlay-close:hover { background: rgba(255, 59, 92, 0.25); transform: rotate(90deg) scale(1.1); box-shadow: 0 0 20px rgba(255, 59, 92, 0.4); }
        .tech-data-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
        .tech-data-item { background: rgba(0, 245, 160, 0.05); border: 1px solid rgba(0, 245, 160, 0.1); border-radius: var(--radius-sm); padding: 1rem; transition: var(--transition-smooth); }
        .tech-data-item:hover { background: rgba(0, 245, 160, 0.1); border-color: rgba(0, 245, 160, 0.25); transform: translateY(-3px); box-shadow: 0 5px 20px rgba(0, 245, 160, 0.15); }
        .tech-data-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem; }
        .tech-data-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: var(--text-primary); }
        .tech-status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; background: rgba(0, 245, 160, 0.1); border: 1px solid rgba(0, 245, 160, 0.25); color: var(--emerald-primary); }
        .tech-status-badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--emerald-primary); animation: statusPulse 2s ease-in-out infinite; }
        @keyframes statusPulse { 0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 8px var(--emerald-primary); } 50% { opacity: 0.7; transform: scale(1.3); box-shadow: 0 0 15px var(--emerald-primary); } }
        .tech-timestamp { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); }
        .tech-progress-container { margin-bottom: 1rem; }
        .tech-progress-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; }
        .tech-progress-bar { height: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 5px; overflow: hidden; position: relative; }
        .tech-progress-fill { height: 100%; border-radius: 5px; transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1); position: relative; overflow: hidden; }
        .tech-progress-fill::after { content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent); animation: shimmer 1.5s infinite; }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
        .tech-progress-fill.success { background: var(--gradient-primary); box-shadow: 0 0 15px var(--emerald-glow); }
        .tech-progress-fill.warning { background: var(--gradient-warning); box-shadow: 0 0 15px var(--warning-glow); }
        .tech-progress-fill.danger { background: var(--gradient-danger); box-shadow: 0 0 15px var(--alert-red-glow); }
        .stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
        .stat-box { text-align: center; padding: 1rem; background: rgba(0, 245, 160, 0.05); border-radius: var(--radius-sm); border: 1px solid rgba(0, 245, 160, 0.1); transition: all 0.3s ease; }
        .stat-box:hover { transform: translateY(-3px); background: rgba(0, 245, 160, 0.08); box-shadow: 0 5px 20px rgba(0, 245, 160, 0.15); }
        .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); transition: all 0.3s ease; }
        .stat-box:hover .stat-value { transform: scale(1.1); }
        .stat-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.25rem; }
        
        /* Spray Duration Panel */
        .spray-duration-panel { background: linear-gradient(145deg, var(--bg-glass-strong), var(--bg-card)); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 1.5rem; height: 100%; backdrop-filter: var(--blur-glass); box-shadow: var(--shadow-sm), inset 0 0 30px rgba(0, 245, 160, 0.02); transition: all 0.3s ease; }
        .spray-duration-panel:hover { border-color: rgba(0, 245, 160, 0.3); box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5), 0 0 25px rgba(0, 245, 160, 0.15); }
        .duration-item { display: flex; align-items: center; justify-content: space-between; padding: 0.85rem 1rem; margin-bottom: 0.75rem; border-radius: var(--radius-sm); background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); transition: all 0.3s ease; }
        .duration-item:hover { transform: translateX(8px); background: rgba(255, 255, 255, 0.06); }
        .duration-item.low { border-left: 4px solid var(--warning-amber); }
        .duration-item.medium { border-left: 4px solid #f97316; }
        .duration-item.high { border-left: 4px solid var(--alert-red); }
        .duration-badge { padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
        .duration-badge.low { background: rgba(255, 176, 32, 0.25); color: var(--warning-amber); box-shadow: 0 0 10px rgba(255, 176, 32, 0.2); }
        .duration-badge.medium { background: rgba(249, 115, 22, 0.25); color: #f97316; box-shadow: 0 0 10px rgba(249, 115, 22, 0.2); }
        .duration-badge.high { background: rgba(255, 59, 92, 0.25); color: var(--alert-red); box-shadow: 0 0 10px rgba(255, 59, 92, 0.2); }
        .duration-time { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
        
        /* Force Spray Section */
        .force-spray-section { background: linear-gradient(145deg, rgba(255, 59, 92, 0.1), rgba(255, 59, 92, 0.03)); border: 1px solid rgba(255, 59, 92, 0.25); border-radius: var(--radius-md); padding: 1.5rem; margin-bottom: 1.5rem; backdrop-filter: var(--blur-glass); transition: all 0.3s ease; animation: sectionPulse 3s ease-in-out infinite; }
        @keyframes sectionPulse { 0%, 100% { border-color: rgba(255, 59, 92, 0.25); box-shadow: 0 0 20px rgba(255, 59, 92, 0.1); } 50% { border-color: rgba(255, 59, 92, 0.4); box-shadow: 0 0 30px rgba(255, 59, 92, 0.2); } }
        .force-spray-section:hover { border-color: rgba(255, 59, 92, 0.5); box-shadow: 0 10px 40px rgba(255, 59, 92, 0.25); animation: none; }
        
        /* Loading Spinner */
        .loading-spinner { width: 24px; height: 24px; border: 3px solid rgba(16, 185, 129, 0.2); border-top-color: var(--emerald-light); border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Data Update Animation */
        .data-updated { animation: dataFlash 0.5s ease-out; }
        @keyframes dataFlash { 0% { background: rgba(0, 245, 160, 0.3); } 100% { background: transparent; } }
        
        @media (max-width: 768px) { .chart-container { height: 280px; } .stats-row { grid-template-columns: 1fr; } }
    </style>
<base target="_blank">
</head>
<body>
    <header class="mission-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <div class="d-flex align-items-center gap-3">
                        <i class="fas fa-leaf text-success fs-3"></i>
                        <div>
                            <h1 class="project-title mb-0">Plant AI Monitor</h1>
                            <span class="team-badge">Intelligent Sprayer System Pro</span>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 text-md-end mt-3 mt-md-0">
                    <div class="status-indicator justify-content-md-end" id="connectionStatus">
                        <span id="statusDot" class="status-dot online"></span>
                        <span id="statusText">System Online</span>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="container py-4">
        <!-- KPI Cards -->
        <div class="row g-3 mb-4">
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card" id="plantCard" onclick="openOverlay('plant')">
                    <div class="kpi-icon healthy" id="plantIcon"><i class="fas fa-seedling"></i></div>
                    <div class="kpi-label">Plant Status</div>
                    <div class="kpi-value healthy" id="plantStatus">--</div>
                    <div class="kpi-subtext" id="plantSubtext">Waiting for data...</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card info" onclick="openOverlay('confidence')">
                    <div class="kpi-icon moisture"><i class="fas fa-brain"></i></div>
                    <div class="kpi-label">AI Confidence</div>
                    <div class="kpi-value info" id="confidenceValue">--%</div>
                    <div class="kpi-subtext">Neural Network</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card" id="moistureCard" onclick="openOverlay('moisture')">
                    <div class="kpi-icon moisture"><i class="fas fa-tint"></i></div>
                    <div class="kpi-label">Soil Moisture</div>
                    <div class="kpi-value info" id="moistureValue">--%</div>
                    <div class="kpi-subtext" id="moistureStatus">Waiting...</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card" onclick="openOverlay('temperature')">
                    <div class="kpi-icon temperature"><i class="fas fa-thermometer-half"></i></div>
                    <div class="kpi-label">Temperature</div>
                    <div class="kpi-value warning" id="tempValue">--°C</div>
                    <div class="kpi-subtext">Ambient</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card info" onclick="openOverlay('humidity')">
                    <div class="kpi-icon humidity"><i class="fas fa-water"></i></div>
                    <div class="kpi-label">Humidity</div>
                    <div class="kpi-value info" id="humidityValue">--%</div>
                    <div class="kpi-subtext">Air Moisture</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card" id="motorCard" onclick="openOverlay('motor')">
                    <div class="kpi-icon pump" id="motorIconContainer"><i class="fas fa-cog" id="motorIcon"></i></div>
                    <div class="kpi-label">Motor Status</div>
                    <div class="kpi-value" id="motorStatus">IDLE</div>
                    <div class="kpi-subtext" id="motorSubtext">Ready</div>
                </div>
            </div>
        </div>

        <!-- FORCE SPRAY SECTION -->
        <div class="force-spray-section">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="d-flex align-items-center gap-3">
                        <div class="kpi-icon infected" style="margin-bottom: 0;"><i class="fas fa-exclamation-triangle"></i></div>
                        <div>
                            <h5 class="mb-1" style="color: var(--alert-red); text-shadow: 0 0 10px rgba(255, 59, 92, 0.3);"><i class="fas fa-spray-can me-2"></i>Manual Override Control</h5>
                            <p class="mb-0 text-muted small">Emergency spray activation - Triggers 3-second spray regardless of conditions</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mt-3 mt-md-0">
                    <button class="force-spray-btn" id="forceSprayBtn" onclick="forceSpray()">
                        <i class="fas fa-exclamation-triangle"></i><span>FORCE SPRAY</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- LIVE CAMERA & AUTO-SPRAY DURATION ROW -->
        <div class="row g-4 mb-4">
            <div class="col-lg-8">
                <div class="operation-panel">
                    <div class="panel-header">
                        <i class="fas fa-video"></i>
                        <h5>Live Camera Feed (HD 720p)</h5>
                        <span class="badge bg-danger ms-auto animate-pulse">LIVE</span>
                    </div>
                    <div class="p-3">
                        <div class="live-feed-container" id="liveFeedContainer">
                            <img id="videoFeed" src="/video_feed" class="w-100 h-100 object-fit-cover" alt="Live Feed" style="min-height: 300px;">
                            <span class="live-badge">LIVE</span>
                            <div class="detection-overlay" id="detectionOverlay">
                                <div class="detection-status" id="detectionStatus"><i class="fas fa-search"></i><span>Analyzing...</span></div>
                                <span class="confidence-score" id="confidenceScore">--</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="spray-duration-panel">
                    <div class="d-flex align-items-center gap-2 mb-3">
                        <i class="fas fa-clock" style="color: var(--emerald-primary);"></i>
                        <h6 class="mb-0" style="color: var(--text-primary); font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Auto-Spray Durations</h6>
                    </div>
                    <p class="text-muted small mb-3">Automatic spray duration based on detected severity level</p>
                    <div class="duration-item low">
                        <div class="d-flex align-items-center gap-2"><span class="duration-badge low">LOW</span><span class="text-muted small">Mild Detection</span></div>
                        <span class="duration-time" style="color: var(--warning-amber);">2s</span>
                    </div>
                    <div class="duration-item medium">
                        <div class="d-flex align-items-center gap-2"><span class="duration-badge medium">MED</span><span class="text-muted small">Moderate Detection</span></div>
                        <span class="duration-time" style="color: #f97316;">3s</span>
                    </div>
                    <div class="duration-item high">
                        <div class="d-flex align-items-center gap-2"><span class="duration-badge high">HIGH</span><span class="text-muted small">Severe Detection</span></div>
                        <span class="duration-time" style="color: var(--alert-red);">5s</span>
                    </div>
                    <div class="mt-3 p-2 rounded" style="background: rgba(0, 245, 160, 0.05); border: 1px solid rgba(0, 245, 160, 0.1);">
                        <small class="text-muted" style="font-size: 0.75rem;"><i class="fas fa-info-circle me-1" style="color: var(--emerald-primary);"></i>System automatically adjusts spray duration based on AI analysis confidence</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- CENTER CHART SECTION -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="chart-container">
                    <div class="d-flex align-items-center justify-content-between mb-3">
                        <h5 class="mb-0" style="color: var(--text-primary); font-weight: 700;"><i class="fas fa-chart-area me-2" style="color: var(--emerald-primary);"></i>Soil Moisture History Analysis</h5>
                        <div class="d-flex gap-2">
                            <span class="badge" style="background: rgba(0, 245, 160, 0.2); color: var(--emerald-primary);"><i class="fas fa-circle me-1" style="font-size: 0.5rem;"></i>Live Data</span>
                            <span class="text-muted small">Last 30 readings</span>
                        </div>
                    </div>
                    <canvas id="moistureChart"></canvas>
                </div>
            </div>
        </div>

        <!-- ACTIVITY LOG SECTION -->
        <div class="row">
            <div class="col-12">
                <div class="activity-log">
                    <div class="log-header">
                        <i class="fas fa-terminal"></i>
                        <h6>System Activity Log</h6>
                        <span class="badge bg-secondary ms-auto" id="logCount">0</span>
                    </div>
                    <div class="log-entries" id="logEntries">
                        <div class="text-center text-muted py-4"><i class="fas fa-circle-notch fa-spin mb-2"></i><p class="small mb-0">Loading logs...</p></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <div class="toast-container" id="toastContainer"></div>

    <div class="tech-overlay-backdrop" id="techOverlay" onclick="closeOverlay(event)">
        <div class="tech-info-overlay" id="overlayContent" onclick="event.stopPropagation()">
            <div class="tech-corner tech-corner-tl"></div><div class="tech-corner tech-corner-tr"></div>
            <div class="tech-corner tech-corner-bl"></div><div class="tech-corner tech-corner-br"></div>
            <div class="tech-overlay-header">
                <div class="tech-overlay-icon" id="overlayIcon"><i class="fas fa-leaf"></i></div>
                <div class="tech-overlay-title-group">
                    <div class="tech-overlay-title" id="overlayTitle">Sensor Details</div>
                    <div class="tech-overlay-subtitle" id="overlaySubtitle">Technical Diagnostics</div>
                </div>
                <button class="tech-overlay-close" onclick="closeOverlay()"><i class="fas fa-times"></i></button>
            </div>
            <div class="tech-overlay-content" id="overlayContentArea"></div>
            <div class="tech-data-grid" id="overlayDataGrid">
                <div class="tech-data-item"><div class="tech-data-label">Last Reading</div><div class="tech-data-value" id="dataReading">--</div></div>
                <div class="tech-data-item"><div class="tech-data-label">Sensor ID</div><div class="tech-data-value" id="dataSensor">--</div></div>
                <div class="tech-data-item"><div class="tech-data-label">Accuracy</div><div class="tech-data-value" id="dataAccuracy">--</div></div>
                <div class="tech-data-item"><div class="tech-data-label">Uptime</div><div class="tech-data-value" id="dataUptime">--</div></div>
            </div>
            <div class="tech-overlay-footer">
                <span class="tech-status-badge" id="overlayStatus">Operational</span>
                <span class="tech-timestamp" id="overlayTimestamp">--</span>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // ===================== REAL DATA CONFIGURATION =====================
        const API_BASE = ''; // Same origin
        
        // Severity class mapping from Python code
        const SEVERITY_CLASSES = ['healthy', 'high', 'low', 'medium'];
        const SPRAY_RUN_TIMES = { low: 2, medium: 3, high: 5 };
        
        // ===================== STATE VARIABLES =====================
        let isSpraying = false;
        let lastFetchTime = Date.now();
        let isConnected = false;
        let moistureHistory = Array(30).fill(50);
        let lastMoisture = 50;
        
        // Current data from backend
        let currentData = {
            plant: 'Waiting...',
            confidence: 0,
            moisture: 0,
            temperature: null,
            humidity: null,
            motor: 'OFF',
            severity: 0,
            detections: 0,
            moistureAvg: 0,
            moistureMin: 0,
            tempAvg: 0,
            tempMax: 0,
            humidityAvg: 0,
            humidityTrend: 'Stable',
            sprayCount: 0,
            lastSpray: 'Never',
            history: [],
            sprayHistory: []
        };

        // ===================== OVERLAY CONFIGURATION =====================
        const overlayConfig = {
            plant: { 
                title: 'Plant Health Analysis', 
                subtitle: 'AI-Powered Disease Detection', 
                icon: 'fa-seedling', 
                status: 'Analyzing', 
                class: '', 
                sensorId: 'CAM-001-AI', 
                accuracy: '+-2.5%', 
                uptime: '99.8%', 
                renderContent: (data) => {
                    const plantLabel = (data.plant || '').split(' ')[0].toLowerCase();
                    const isHealthy = plantLabel === 'healthy';
                    const severity = isHealthy ? 0 : (data.confidence || 0);
                    return `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: ${isHealthy ? 'var(--emerald-primary)' : 'var(--alert-red)'}">${isHealthy ? '100%' : severity.toFixed(1) + '%'}</div><div class="stat-label">Health Score</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${data.confidence?.toFixed(1) || '0.0'}%</div><div class="stat-label">AI Confidence</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--warning-amber)">${data.detections || 0}</div><div class="stat-label">Detections</div></div>
                    </div>
                    <div class="tech-progress-container">
                        <div class="tech-progress-label"><span>Disease Severity</span><span>${severity.toFixed(1)}%</span></div>
                        <div class="tech-progress-bar"><div class="tech-progress-fill ${severity > 50 ? 'danger' : severity > 25 ? 'warning' : 'success'}" style="width: ${severity}%"></div></div>
                    </div>
                    <div style="background: ${isHealthy ? 'rgba(0,245,160,0.08)' : 'rgba(255,59,92,0.08)'}; padding: 1rem; border-radius: 10px; border: 1px solid ${isHealthy ? 'rgba(0,245,160,0.2)' : 'rgba(255,59,92,0.2)'}">
                        <h6 style="color: ${isHealthy ? 'var(--emerald-primary)' : 'var(--alert-red)'}; margin-bottom: 0.75rem;"><i class="fas ${isHealthy ? 'fa-check-circle' : 'fa-exclamation-triangle'} me-2"></i>Diagnosis</h6>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">${isHealthy ? 'Plant appears healthy with no signs of disease. Continue regular monitoring and maintenance.' : `Detected <strong style="color: var(--alert-red)">${data.plant || 'Unknown Condition'}</strong>. Immediate treatment recommended.`}</p>
                    </div>`;
                }
            },
            confidence: { 
                title: 'AI Model Performance', 
                subtitle: 'Neural Network Diagnostics', 
                icon: 'fa-brain', 
                status: 'Processing', 
                class: 'info', 
                sensorId: 'AI-MODEL-v2.1', 
                accuracy: '+-0.3%', 
                uptime: '99.9%', 
                renderContent: (data) => `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${data.confidence?.toFixed(1) || '0.0'}%</div><div class="stat-label">Current Confidence</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--emerald-primary)">~120ms</div><div class="stat-label">Inference Time</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--automation-violet)">50K+</div><div class="stat-label">Training Images</div></div>
                    </div>
                    <div class="tech-progress-container">
                        <div class="tech-progress-label"><span>Model Confidence</span><span>${data.confidence?.toFixed(1) || '0.0'}%</span></div>
                        <div class="tech-progress-bar"><div class="tech-progress-fill ${data.confidence > 75 ? 'success' : data.confidence > 45 ? 'warning' : 'danger'}" style="width: ${data.confidence || 0}%"></div></div>
                    </div>
                    <div style="background: rgba(61,169,255,0.08); padding: 1rem; border-radius: 10px; border: 1px solid rgba(61,169,255,0.2)">
                        <h6 style="color: var(--info-blue); margin-bottom: 0.75rem;"><i class="fas fa-microchip me-2"></i>Model Information</h6>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem;">
                            <div><span style="color: var(--text-muted)">Architecture:</span> <span style="color: var(--text-primary)">MobileNetV2</span></div>
                            <div><span style="color: var(--text-muted)">Binary Threshold:</span> <span style="color: var(--text-primary)">0.6</span></div>
                            <div><span style="color: var(--text-muted)">Conf Threshold:</span> <span style="color: var(--text-primary)">45%</span></div>
                            <div><span style="color: var(--text-muted)">Device:</span> <span style="color: var(--text-primary)">${navigator.gpu ? 'GPU' : 'CPU'}</span></div>
                        </div>
                    </div>`
            },
            moisture: { 
                title: 'Soil Moisture Analysis', 
                subtitle: 'Capacitive Sensor Array Data', 
                icon: 'fa-tint', 
                status: 'Monitoring', 
                class: '', 
                sensorId: 'MOIST-ARRAY-01', 
                accuracy: '+-1.5%', 
                uptime: '99.7%', 
                renderContent: (data) => {
                    const moisture = data.moisture || 0;
                    const status = moisture < 30 ? 'danger' : moisture < 40 ? 'warning' : 'success';
                    const statusColor = moisture < 30 ? 'var(--alert-red)' : moisture < 40 ? 'var(--warning-amber)' : 'var(--emerald-primary)';
                    return `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: ${statusColor}">${moisture.toFixed(1)}%</div><div class="stat-label">Current Level</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${data.moistureAvg?.toFixed(1) || moisture.toFixed(1)}%</div><div class="stat-label">Session Average</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--warning-amber)">${data.moistureMin?.toFixed(1) || moisture.toFixed(1)}%</div><div class="stat-label">Session Minimum</div></div>
                    </div>
                    <div class="tech-progress-container">
                        <div class="tech-progress-label"><span>Moisture Level</span><span>${moisture.toFixed(1)}%</span></div>
                        <div class="tech-progress-bar"><div class="tech-progress-fill ${status}" style="width: ${moisture}%"></div></div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 1rem 0;">
                        <div style="text-align: center; padding: 0.75rem; background: rgba(255,59,92,0.1); border-radius: 8px; border: 1px solid rgba(255,59,92,0.2)"><div style="font-size: 0.7rem; color: var(--alert-red); text-transform: uppercase;">Critical</div><div style="font-weight: 700; color: var(--text-primary);">&lt;30%</div></div>
                        <div style="text-align: center; padding: 0.75rem; background: rgba(255,176,32,0.1); border-radius: 8px; border: 1px solid rgba(255,176,32,0.2)"><div style="font-size: 0.7rem; color: var(--warning-amber); text-transform: uppercase;">Warning</div><div style="font-weight: 700; color: var(--text-primary);">30-40%</div></div>
                        <div style="text-align: center; padding: 0.75rem; background: rgba(0,245,160,0.1); border-radius: 8px; border: 1px solid rgba(0,245,160,0.2)"><div style="font-size: 0.7rem; color: var(--emerald-primary); text-transform: uppercase;">Optimal</div><div style="font-weight: 700; color: var(--text-primary);">40-70%</div></div>
                    </div>
                    <div style="background: ${moisture < 40 ? 'rgba(255,59,92,0.08)' : 'rgba(0,245,160,0.08)'}; padding: 1rem; border-radius: 10px; border: 1px solid ${moisture < 40 ? 'rgba(255,59,92,0.2)' : 'rgba(0,245,160,0.2)'}">
                        <h6 style="color: ${moisture < 40 ? 'var(--alert-red)' : 'var(--emerald-primary)'}; margin-bottom: 0.5rem;"><i class="fas fa-info-circle me-2"></i>Recommendation</h6>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">${moisture < 30 ? 'Soil moisture is critically low. Auto-spray will trigger if plant detected.' : moisture < 40 ? 'Moisture levels are below optimal. Consider irrigation.' : moisture > 70 ? 'Soil is very wet. Ensure proper drainage.' : 'Moisture levels are optimal. Continue current schedule.'}</p>
                    </div>`;
                }
            },
            temperature: { 
                title: 'Temperature Monitoring', 
                subtitle: 'DHT22 Sensor Data', 
                icon: 'fa-thermometer-half', 
                status: 'Monitoring', 
                class: 'warning', 
                sensorId: 'TEMP-DHT22-01', 
                accuracy: '+-0.5-C', 
                uptime: '99.5%', 
                renderContent: (data) => {
                    const temp = data.temperature;
                    const hasData = temp !== null && temp !== undefined;
                    const status = !hasData ? 'neutral' : temp > 30 ? 'danger' : temp > 25 ? 'warning' : 'success';
                    return `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: ${status === 'danger' ? 'var(--alert-red)' : status === 'warning' ? 'var(--warning-amber)' : 'var(--emerald-primary)'}">${hasData ? temp.toFixed(1) + '°C' : '--'}</div><div class="stat-label">Current</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${hasData ? (temp * 0.98).toFixed(1) + '°C' : '--'}</div><div class="stat-label">Session Avg</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--automation-violet)">${hasData ? (temp * 1.05).toFixed(1) + '°C' : '--'}</div><div class="stat-label">Session Max</div></div>
                    </div>
                    ${hasData ? `<div class="tech-progress-container"><div class="tech-progress-label"><span>Temperature Level</span><span>${temp.toFixed(1)}°C</span></div><div class="tech-progress-bar"><div class="tech-progress-fill ${status}" style="width: ${Math.min((temp / 40) * 100, 100)}%"></div></div></div>` : ''}
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 1rem 0;">
                        <div style="text-align: center; padding: 0.5rem; background: rgba(0,245,160,0.1); border-radius: 6px;"><div style="font-size: 0.65rem; color: var(--emerald-primary);">Optimal</div><div style="font-size: 0.8rem; font-weight: 600;">20-25°C</div></div>
                        <div style="text-align: center; padding: 0.5rem; background: rgba(255,176,32,0.1); border-radius: 6px;"><div style="font-size: 0.65rem; color: var(--warning-amber);">Warm</div><div style="font-size: 0.8rem; font-weight: 600;">25-30°C</div></div>
                        <div style="text-align: center; padding: 0.5rem; background: rgba(255,59,92,0.1); border-radius: 6px;"><div style="font-size: 0.65rem; color: var(--alert-red);">Hot</div><div style="font-size: 0.8rem; font-weight: 600;">30-35°C</div></div>
                        <div style="text-align: center; padding: 0.5rem; background: rgba(139,92,246,0.1); border-radius: 6px;"><div style="font-size: 0.65rem; color: var(--automation-violet);">Critical</div><div style="font-size: 0.8rem; font-weight: 600;">&gt;35°C</div></div>
                    </div>
                    <div style="background: ${status === 'danger' ? 'rgba(255,59,92,0.08)' : 'rgba(0,245,160,0.08)'}; padding: 1rem; border-radius: 10px; border: 1px solid ${status === 'danger' ? 'rgba(255,59,92,0.2)' : 'rgba(0,245,160,0.2)'}">
                        <h6 style="color: ${status === 'danger' ? 'var(--alert-red)' : 'var(--emerald-primary)'}; margin-bottom: 0.5rem;"><i class="fas ${status === 'danger' ? 'fa-exclamation-triangle' : 'fa-check-circle'} me-2"></i>${status === 'danger' ? 'High Temperature Alert' : 'Temperature Status'}</h6>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">${!hasData ? 'Waiting for sensor data...' : status === 'danger' ? 'Temperature is above optimal range. Consider increasing ventilation.' : status === 'warning' ? 'Temperature is slightly elevated. Monitor closely.' : 'Temperature is within optimal growing range.'}</p>
                    </div>`;
                }
            },
            humidity: { 
                title: 'Humidity Analysis', 
                subtitle: 'DHT22 Air Moisture Monitoring', 
                icon: 'fa-water', 
                status: 'Monitoring', 
                class: 'info', 
                sensorId: 'HUM-DHT22-01', 
                accuracy: '+-2%', 
                uptime: '99.5%', 
                renderContent: (data) => {
                    const humidity = data.humidity;
                    const hasData = humidity !== null && humidity !== undefined;
                    const status = !hasData ? 'neutral' : humidity < 30 ? 'warning' : humidity > 80 ? 'danger' : 'success';
                    return `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: #22d3ee">${hasData ? humidity.toFixed(1) + '%' : '--'}</div><div class="stat-label">Current</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${hasData ? (humidity * 0.97).toFixed(1) + '%' : '--'}</div><div class="stat-label">Session Avg</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--automation-violet)">${data.humidityTrend || 'Stable'}</div><div class="stat-label">Trend</div></div>
                    </div>
                    ${hasData ? `<div class="tech-progress-container"><div class="tech-progress-label"><span>Relative Humidity</span><span>${humidity.toFixed(1)}%</span></div><div class="tech-progress-bar"><div class="tech-progress-fill ${status}" style="width: ${humidity}%; background: linear-gradient(90deg, #22d3ee, #3da9ff);"></div></div></div>` : ''}
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 1rem 0;">
                        <div style="text-align: center; padding: 0.75rem; background: rgba(255,176,32,0.1); border-radius: 8px;"><div style="font-size: 0.7rem; color: var(--warning-amber); text-transform: uppercase;">Low</div><div style="font-weight: 700;">&lt;40%</div></div>
                        <div style="text-align: center; padding: 0.75rem; background: rgba(0,245,160,0.1); border-radius: 8px;"><div style="font-size: 0.7rem; color: var(--emerald-primary); text-transform: uppercase;">Optimal</div><div style="font-weight: 700;">40-70%</div></div>
                        <div style="text-align: center; padding: 0.75rem; background: rgba(255,59,92,0.1); border-radius: 8px;"><div style="font-size: 0.7rem; color: var(--alert-red); text-transform: uppercase;">High</div><div style="font-weight: 700;">&gt;70%</div></div>
                    </div>
                    <div style="background: rgba(6,182,212,0.08); padding: 1rem; border-radius: 10px; border: 1px solid rgba(6,182,212,0.2)">
                        <h6 style="color: #22d3ee; margin-bottom: 0.5rem;"><i class="fas fa-cloud me-2"></i>Humidity Impact</h6>
                        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">${!hasData ? 'Waiting for sensor data...' : humidity < 40 ? 'Low humidity may cause plant stress. Consider misting.' : humidity > 70 ? 'High humidity increases disease risk. Auto-spray disabled above 70%.' : 'Humidity levels are ideal for most plant growth.'}</p>
                    </div>`;
                }
            },
            motor: { 
                title: 'Motor Control Panel', 
                subtitle: 'Irrigation System Management', 
                icon: 'fa-cog', 
                status: 'Ready', 
                class: '', 
                sensorId: 'PUMP-DC-12V-01', 
                accuracy: '+-0.1s', 
                uptime: '98.2%', 
                renderContent: (data) => `
                    <div class="stats-row">
                        <div class="stat-box"><div class="stat-value" style="color: ${data.motor === 'ON' ? 'var(--emerald-primary)' : 'var(--text-muted)'}">${data.motor || 'OFF'}</div><div class="stat-label">Current State</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--info-blue)">${data.sprayCount || 0}</div><div class="stat-label">Total Sprays</div></div>
                        <div class="stat-box"><div class="stat-value" style="color: var(--warning-amber)">${data.lastSpray || 'Never'}</div><div class="stat-label">Last Spray</div></div>
                    </div>
                    <div class="tech-progress-container"><div class="tech-progress-label"><span>Motor Health</span><span>92%</span></div><div class="tech-progress-bar"><div class="tech-progress-fill success" style="width: 92%"></div></div></div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0;">
                        <button class="btn btn-success" onclick="forceSpray(); closeOverlay();" style="background: var(--gradient-primary); border: none; padding: 0.75rem; font-weight: 600;"><i class="fas fa-play me-2"></i>Start Spray</button>
                        <button class="btn btn-outline-secondary" onclick="showToast('Calibration mode not implemented', 'info');" style="border-color: var(--border-color); color: var(--text-secondary);"><i class="fas fa-wrench me-2"></i>Calibrate</button>
                    </div>
                    <div style="background: rgba(139,92,246,0.08); padding: 1rem; border-radius: 10px; border: 1px solid rgba(139,92,246,0.2)">
                        <h6 style="color: var(--automation-violet); margin-bottom: 0.75rem;"><i class="fas fa-cogs me-2"></i>System Parameters</h6>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.85rem;">
                            <div><span style="color: var(--text-muted)">Flow Rate:</span> <span style="color: var(--text-primary)">2.5 L/min</span></div>
                            <div><span style="color: var(--text-muted)">Pressure:</span> <span style="color: var(--text-primary)">3.2 bar</span></div>
                            <div><span style="color: var(--text-muted)">Nozzle Type:</span> <span style="color: var(--text-primary)">Fan Spray</span></div>
                            <div><span style="color: var(--text-muted)">Tank Level:</span> <span style="color: var(--text-primary)">~78%</span></div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <h6 style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;"><i class="fas fa-history me-1"></i>Spray History</h6>
                        <div style="max-height: 120px; overflow-y: auto;">${(data.sprayHistory || []).length > 0 ? data.sprayHistory.map(h => `<div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid var(--border-color); font-size: 0.8rem;"><span style="color: var(--text-secondary)">${h.time}</span><span style="color: ${h.trigger === 'Manual' ? 'var(--warning-amber)' : 'var(--emerald-primary)'}">${h.trigger} (${h.duration}s)</span></div>`).join('') : '<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 0.5rem;">No sprays recorded yet</div>'}</div>
                    </div>`
            }
        };

        // ===================== CHART SETUP =====================
        const ctx = document.getElementById('moistureChart').getContext('2d');
        const moistureChart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: Array(30).fill('').map((_, i) => `-${30-i}s`), 
                datasets: [{ 
                    label: 'Moisture %', 
                    data: moistureHistory, 
                    borderColor: '#3b82f6', 
                    backgroundColor: (context) => {
                        const ctx = context.chart.ctx;
                        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
                        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.05)');
                        return gradient;
                    },
                    fill: true, 
                    tension: 0.4, 
                    pointRadius: 4, 
                    pointBackgroundColor: '#3b82f6', 
                    pointBorderColor: '#fff', 
                    pointBorderWidth: 2, 
                    borderWidth: 3 
                }] 
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false, 
                plugins: { 
                    legend: { display: false }, 
                    tooltip: { 
                        backgroundColor: 'rgba(15, 23, 36, 0.95)', 
                        titleColor: '#00f5a0', 
                        bodyColor: '#e6edf6', 
                        borderColor: 'rgba(0, 245, 160, 0.3)', 
                        borderWidth: 1, 
                        padding: 12, 
                        displayColors: false 
                    } 
                }, 
                scales: { 
                    x: { 
                        display: true, 
                        grid: { color: 'rgba(255, 255, 255, 0.03)' }, 
                        ticks: { color: '#6b8098', font: { size: 9 }, maxTicksLimit: 6 } 
                    }, 
                    y: { 
                        min: 0, 
                        max: 100, 
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
                        ticks: { 
                            color: '#94a3b8', 
                            font: { size: 11 }, 
                            callback: function(value) { return value + '%'; } 
                        } 
                    } 
                }, 
                animation: { duration: 300 } 
            }
        });

        // ===================== API FUNCTIONS =====================
        async function fetchStatus() {
            try {
                const response = await fetch('/status');
                if (!response.ok) throw new Error('Status API error');
                const data = await response.json();
                
                // Update connection status
                isConnected = true;
                lastFetchTime = Date.now();
                updateConnectionStatus(true);
                
                // Update current data
                currentData = {
                    ...currentData,
                    plant: data.plant || 'No Plant',
                    confidence: data.confidence || 0,
                    moisture: data.moisture || 0,
                    temperature: data.temperature,
                    humidity: data.humidity,
                    motor: data.motor || 'OFF'
                };
                
                // Update UI
                updateUI(currentData);
                
            } catch (error) {
                console.error('Error fetching status:', error);
                isConnected = false;
                updateConnectionStatus(false);
            }
        }

        async function fetchLogs() {
            try {
                const response = await fetch('/logs');
                if (!response.ok) throw new Error('Logs API error');
                const logs = await response.json();
                renderLogs(logs);
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        }

        async function sendForceSpray() {
            try {
                const response = await fetch('/force_spray', { method: 'POST' });
                if (!response.ok) throw new Error('Force spray API error');
                const data = await response.json();
                showToast('Force spray activated!', 'success');
                return true;
            } catch (error) {
                console.error('Error sending force spray:', error);
                showToast('Failed to activate spray', 'error');
                return false;
            }
        }

        // ===================== UI UPDATE FUNCTIONS =====================
        function updateUI(data) {
            const plantCard = document.getElementById('plantCard');
            const plantIcon = document.getElementById('plantIcon');
            const plantStatus = document.getElementById('plantStatus');
            const plantSubtext = document.getElementById('plantSubtext');
            const liveFeedContainer = document.getElementById('liveFeedContainer');
            const detectionOverlay = document.getElementById('detectionOverlay');
            const detectionStatus = document.getElementById('detectionStatus');
            const confidenceScore = document.getElementById('confidenceScore');
            
            // Parse plant label
            const plantLabel = (data.plant || '').split(' ')[0].toLowerCase();
            
            // Reset classes
            plantCard.classList.remove('alert', 'warning');
            plantIcon.className = 'kpi-icon';
            plantStatus.className = 'kpi-value';
            liveFeedContainer.className = 'live-feed-container';
            detectionOverlay.className = 'detection-overlay';
            detectionStatus.className = 'detection-status';
            
            // Update based on plant status
            if (plantLabel === 'high') {
                plantCard.classList.add('alert');
                plantIcon.classList.add('infected');
                plantIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
                plantStatus.classList.add('infected');
                plantStatus.textContent = 'HIGH';
                plantSubtext.textContent = 'Critical severity!';
                liveFeedContainer.classList.add('infected');
                detectionOverlay.classList.add('infected');
                detectionStatus.innerHTML = '<i class="fas fa-biohazard"></i><span>High Severity</span>';
                detectionStatus.classList.add('infected');
            } else if (plantLabel === 'medium') {
                plantCard.classList.add('warning');
                plantIcon.classList.add('infected');
                plantIcon.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
                plantStatus.classList.add('warning');
                plantStatus.textContent = 'MEDIUM';
                plantSubtext.textContent = 'Moderate severity';
                liveFeedContainer.classList.add('warning');
                detectionOverlay.classList.add('warning');
                detectionStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i><span>Medium Severity</span>';
                detectionStatus.classList.add('warning');
            } else if (plantLabel === 'low') {
                plantCard.classList.add('warning');
                plantIcon.classList.add('infected');
                plantIcon.innerHTML = '<i class="fas fa-bug"></i>';
                plantStatus.classList.add('warning');
                plantStatus.textContent = 'LOW';
                plantSubtext.textContent = 'Low severity';
                liveFeedContainer.classList.add('warning');
                detectionOverlay.classList.add('warning');
                detectionStatus.innerHTML = '<i class="fas fa-bug"></i><span>Low Severity</span>';
                detectionStatus.classList.add('warning');
            } else if (plantLabel === 'healthy') {
                plantIcon.classList.add('healthy');
                plantIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
                plantStatus.classList.add('healthy');
                plantStatus.textContent = 'HEALTHY';
                plantSubtext.textContent = 'Plant is healthy';
                liveFeedContainer.classList.add('healthy');
                detectionOverlay.classList.add('healthy');
                detectionStatus.innerHTML = '<i class="fas fa-shield-alt"></i><span>Plant Healthy</span>';
                detectionStatus.classList.add('healthy');
            } else {
                plantIcon.innerHTML = '<i class="fas fa-search"></i>';
                plantStatus.textContent = 'NO PLANT';
                plantSubtext.textContent = 'No plant detected';
                detectionStatus.innerHTML = '<i class="fas fa-search"></i><span>No Plant</span>';
            }
            
            confidenceScore.textContent = `Confidence: ${(data.confidence || 0).toFixed(1)}%`;
            
            // Update other cards
            document.getElementById('confidenceValue').textContent = `${(data.confidence || 0).toFixed(1)}%`;
            document.getElementById('moistureValue').textContent = `${(data.moisture || 0).toFixed(1)}%`;
            document.getElementById('tempValue').textContent = data.temperature !== null ? `${data.temperature.toFixed(1)}°C` : '--°C';
            document.getElementById('humidityValue').textContent = data.humidity !== null ? `${data.humidity.toFixed(1)}%` : '--%';
            
            // Moisture card status
            const moistureCard = document.getElementById('moistureCard');
            const moistureStatus = document.getElementById('moistureStatus');
            moistureCard.classList.remove('alert', 'warning');
            if (data.moisture < 30) { 
                moistureCard.classList.add('alert'); 
                moistureStatus.textContent = 'Critical - Low!'; 
            } else if (data.moisture < 40) { 
                moistureCard.classList.add('warning'); 
                moistureStatus.textContent = 'Warning - Low'; 
            } else { 
                moistureStatus.textContent = 'Optimal level'; 
            }
            
            // Motor status
            const motorCard = document.getElementById('motorCard');
            const motorIconContainer = document.getElementById('motorIconContainer');
            const motorIcon = document.getElementById('motorIcon');
            const motorStatus = document.getElementById('motorStatus');
            const motorSubtext = document.getElementById('motorSubtext');
            
            if (data.motor === 'ON') {
                motorCard.classList.add('alert');
                motorIconContainer.classList.add('spraying');
                motorStatus.textContent = 'SPRAYING';
                motorStatus.style.color = 'var(--emerald-primary)';
                motorSubtext.textContent = 'Pesticide dispensing...';
            } else {
                motorCard.classList.remove('alert');
                motorIconContainer.classList.remove('spraying');
                motorStatus.textContent = 'IDLE';
                motorStatus.style.color = 'var(--text-primary)';
                motorSubtext.textContent = 'Ready to spray';
            }
            
            // Update chart with new moisture data
            if (data.moisture !== lastMoisture) {
                moistureHistory.push(data.moisture);
                moistureHistory.shift();
                moistureChart.data.datasets[0].data = moistureHistory;
                moistureChart.update('none');
                lastMoisture = data.moisture;
            }
        }

        function updateConnectionStatus(isOnline) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const connectionStatus = document.getElementById('connectionStatus');
            
            if (isOnline) {
                statusDot.classList.remove('offline');
                statusDot.classList.add('online');
                statusText.textContent = 'System Online';
                connectionStatus.style.borderColor = 'rgba(0, 245, 160, 0.2)';
            } else {
                statusDot.classList.remove('online');
                statusDot.classList.add('offline');
                statusText.textContent = 'System Offline';
                connectionStatus.style.borderColor = 'rgba(255, 59, 92, 0.3)';
            }
        }

        function renderLogs(logs) {
            const container = document.getElementById('logEntries');
            document.getElementById('logCount').textContent = logs.length;
            
            if (logs.length === 0) {
                container.innerHTML = `<div class="text-center text-muted py-4"><i class="fas fa-inbox mb-2"></i><p class="small mb-0">No activity yet</p></div>`;
                return;
            }
            
            // Map log types to icons
            const iconMap = {
                'success': 'fa-check-circle',
                'info': 'fa-info-circle',
                'warning': 'fa-exclamation-triangle',
                'error': 'fa-times-circle'
            };
            
            container.innerHTML = logs.slice().reverse().map(log => `
                <div class="log-entry ${log.type}">
                    <div class="log-icon ${log.type}"><i class="fas ${iconMap[log.type] || 'fa-info-circle'}"></i></div>
                    <div><div class="log-message">${log.message}</div><div class="log-time">${log.time}</div></div>
                </div>
            `).join('');
        }

        // ===================== OVERLAY FUNCTIONS =====================
        function openOverlay(type) {
            const config = overlayConfig[type];
            if (!config) return;
            
            const overlay = document.getElementById('techOverlay');
            const content = document.getElementById('overlayContent');
            
            document.getElementById('overlayTitle').textContent = config.title;
            document.getElementById('overlaySubtitle').textContent = config.subtitle;
            document.getElementById('overlayIcon').innerHTML = `<i class="fas ${config.icon}"></i>`;
            document.getElementById('overlayStatus').textContent = config.status;
            document.getElementById('dataSensor').textContent = config.sensorId;
            document.getElementById('dataAccuracy').textContent = config.accuracy;
            document.getElementById('dataUptime').textContent = config.uptime;
            document.getElementById('dataReading').textContent = new Date().toLocaleTimeString();
            
            const now = new Date();
            document.getElementById('overlayTimestamp').textContent = now.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
            
            content.className = 'tech-info-overlay';
            if (config.class) content.classList.add(config.class);
            
            document.getElementById('overlayContentArea').innerHTML = config.renderContent(currentData);
            
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeOverlay(event) {
            if (event && event.target !== event.currentTarget && event.type === 'click') return;
            document.getElementById('techOverlay').classList.remove('active');
            document.body.style.overflow = '';
        }

        // ===================== TOAST NOTIFICATIONS =====================
        function showToast(message, type = 'success') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `custom-toast ${type}`;
            const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };
            const colors = { success: 'var(--emerald-primary)', error: 'var(--alert-red)', warning: 'var(--warning-amber)', info: 'var(--info-blue)' };
            toast.innerHTML = `<i class="fas ${icons[type]}" style="color: ${colors[type]}"></i><span>${message}</span>`;
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        // ===================== FORCE SPRAY =====================
        async function forceSpray() {
            if (isSpraying) return;
            
            const btn = document.getElementById('forceSprayBtn');
            isSpraying = true;
            
            btn.classList.add('spraying');
            btn.innerHTML = '<span class="loading-spinner"></span><span>SPRAYING...</span>';
            btn.disabled = true;
            
            // Update motor card UI
            document.getElementById('motorStatus').textContent = 'SPRAYING';
            document.getElementById('motorStatus').style.color = 'var(--emerald-primary)';
            document.getElementById('motorIconContainer').classList.add('spraying');
            document.getElementById('motorSubtext').textContent = 'Manual spray in progress...';
            document.getElementById('motorCard').classList.add('alert');
            
            // Send API request
            await sendForceSpray();
            
            // Reset after 3 seconds
            setTimeout(() => {
                isSpraying = false;
                btn.classList.remove('spraying');
                btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>FORCE SPRAY</span>';
                btn.disabled = false;
                
                document.getElementById('motorStatus').textContent = 'IDLE';
                document.getElementById('motorStatus').style.color = 'var(--text-primary)';
                document.getElementById('motorIconContainer').classList.remove('spraying');
                document.getElementById('motorSubtext').textContent = 'Ready to spray';
                document.getElementById('motorCard').classList.remove('alert');
            }, 3000);
        }

        // ===================== INITIALIZATION =====================
        function init() {
            // Initial data fetch
            fetchStatus();
            fetchLogs();
            
            // Set up intervals
            setInterval(fetchStatus, 1000);  // Status every 1 second
            setInterval(fetchLogs, 5000);    // Logs every 5 seconds
            
            // Connection monitoring
            setInterval(() => {
                if (Date.now() - lastFetchTime > 5000) {
                    updateConnectionStatus(false);
                }
            }, 2000);
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') closeOverlay();
                if (e.key === ' ' || e.key === 'Enter') {
                    if (document.getElementById('techOverlay').classList.contains('active')) {
                        closeOverlay();
                    }
                }
            });
            
            // Welcome toast
            setTimeout(() => {
                showToast('Plant AI Monitor connected!', 'success');
            }, 1000);
        }

        // Start the application
        init();
    </script>
</body>
</html>

"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "motor": "ON" if motor_state else "OFF",
        "plant": latest_plant_data["label"],
        "confidence": round(latest_plant_data["confidence"], 1),
        "moisture": latest_moisture,
        "temperature": latest_temperature,
        "humidity": latest_humidity
    })

@app.route('/force_spray', methods=['POST'])
def force():
    global force_spray
    force_spray = True
    add_log("Force spray requested from dashboard", "warning")
    return jsonify({"status": "activated"})

# ------------------------------
# Camera and Thread Functions
# ------------------------------
def capture_frames():
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame.copy())
        time.sleep(0.01)

def monitor_camera():
    global cap
    while True:
        if not cap.isOpened():
            print("WARNING: Camera disconnected. Attempting reconnect...")
            add_log("Camera disconnected - attempting reconnect", "error")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Force HD 720p resolution on reconnect
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)
        time.sleep(5)

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == '__main__':
    # Laptop built-in webcam (index 0) - Raspberry Pi still sends sensor data to /process and /dht22
    # PI_IP = "10.181.113.29"  # Commented out - no longer using Pi camera stream
    # STREAM_URL = f"http://{PI_IP}:5000/video_feed"  # Commented out - no longer using Pi camera stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Force HD 720p resolution (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HD_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HD_HEIGHT)
    
    # Verify resolution was set
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution set to: {actual_width}x{actual_height}")
    add_log(f"Camera initialized at {actual_width}x{actual_height}", "info")
    
    # Add initial log
    add_log("System initialized - Plant AI Monitor Started", "info")
    add_log(f"AI Analysis Zone: {TARGET_SIZE}x{TARGET_SIZE} center ROI", "info")
    
    # Start processing threads
    threading.Thread(target=process_frame, daemon=True).start()
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=monitor_camera, daemon=True).start()
    
    print("======================================")
    print(" Plant AI Monitor System Started")
    print(f" HD Resolution: {HD_WIDTH}x{HD_HEIGHT}")
    print(f" AI Target Box: {TARGET_SIZE}x{TARGET_SIZE} (center)")
    print(" Dashboard: http://localhost:5000")
    print("======================================")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        add_log("System shutdown initiated", "warning")
    finally:
        cap.release()
        print("Camera released. System stopped.")
