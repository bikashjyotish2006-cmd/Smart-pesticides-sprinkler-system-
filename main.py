import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time
from flask import Flask, request, jsonify, render_template_string, Response
from datetime import datetime


# -------------------------------------------------------
# GPU Setup
# Just telling tensorflow to not grab all GPU memory at once
# -------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU found! Using {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, running on CPU.")


# -------------------------------------------------------
# Load both AI models
# First model checks if there's a plant or not
# Second model tells the disease severity
# -------------------------------------------------------
BINARY_MODEL_PATH = "plant_vs_nonplant_mobilenetv2_final.h5"
SEVERITY_MODEL_PATH = "suraj_chand_severity_mobilenetv2_optimized_final_ooooop.h5"

try:
    binary_model = load_model(BINARY_MODEL_PATH, compile=False)
    severity_model = load_model(SEVERITY_MODEL_PATH, compile=False)
    print("Both models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# The severity labels our model was trained on
severity_classes = ['healthy', 'high', 'low', 'medium']

# Image size that MobileNetV2 expects
IMG_SIZE = (224, 224)

# If binary prediction is above this, we consider it a plant
PLANT_DETECTION_THRESHOLD = 0.99

# Minimum confidence to show a severity result
MIN_CONFIDENCE = 35.0

# If confidence is above this, it's a strong detection
HIGH_CONFIDENCE = 75.0

# The center crop box we feed to the model (in pixels)
CROP_SIZE = 300

# HD resolution for display
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Motor spray time depends on how bad the disease is
spray_durations = {
    'low': 2,    # mild infection, short spray
    'medium': 3, # moderate, medium spray
    'high': 5    # severe, long spray
}

# Thresholds for deciding if we should water the plant
MOISTURE_LIMIT = 40.0   # soil moisture below 40% → dry
HUMIDITY_LIMIT = 70.0   # air humidity below 70%
TEMP_LIMIT = 30.0       # temperature below 30°C


# -------------------------------------------------------
# Global state variables
# These get updated from different threads so we need them global
# -------------------------------------------------------
motor_running = False
motor_started_at = 0
motor_run_for = 0

# For smoothing out prediction flickering
HISTORY_SIZE = 20
SMOOTHING_ALPHA = 0.1
HYSTERESIS = 5
prediction_history = deque(maxlen=HISTORY_SIZE)
plant_currently_visible = False
frames_without_plant = 0

# Flask app
app = Flask(__name__)

# Latest command to send to the Pi
current_motor_command = "STOP"
command_duration = 0

# Latest sensor readings (updated when Pi sends data)
latest_moisture = 0.0
latest_temperature = None
latest_humidity = None

# Current plant detection result
latest_plant = {"label": "No Plant Detected", "confidence": 0.0}

# Emergency spray flag (set to True when user clicks force spray)
force_spray_now = False

# Activity log (only keep last 50 entries)
activity_log = deque(maxlen=50)

# Frame buffers for the producer-consumer threading pattern
raw_frame_queue = deque(maxlen=1)
processed_frame_queue = deque(maxlen=1)

# For freeze mode (we freeze the frame when spray triggers)
freeze_active = False
saved_frame = None
spray_in_progress = False
spray_started_at = 0
spray_lasts_for = 0
manual_spray_prompt = ""


def log_event(message, level="info"):
    """Just adds a timestamped message to our log list."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {
        "time": ts,
        "message": message,
        "type": level  # can be: info, warning, success, error
    }
    activity_log.append(entry)


def get_stable_prediction():
    """
    Averages recent predictions to avoid flickery output.
    Uses exponential smoothing and hysteresis so it doesn't flip rapidly.
    """
    global plant_currently_visible, frames_without_plant

    if not prediction_history:
        return "No Plant Detected", (0, 0, 255), 0.0

    # Exponential moving average of confidence values
    smoothed = 0.0
    for i, (_, conf) in enumerate(prediction_history):
        if i == 0:
            smoothed = conf
        else:
            smoothed = SMOOTHING_ALPHA * conf + (1 - SMOOTHING_ALPHA) * smoothed

    # Find the most common label in recent history
    valid_labels = [lbl for lbl, _ in prediction_history if lbl != "No Plant Detected"]
    if valid_labels:
        most_common = max(set(valid_labels), key=valid_labels.count)
    else:
        most_common = "No Plant Detected"

    if smoothed >= MIN_CONFIDENCE:
        plant_currently_visible = True
        frames_without_plant = 0
        return most_common, (0, 255, 0), smoothed
    else:
        frames_without_plant += 1
        # Hysteresis: keep showing plant for a few frames before saying "no plant"
        if plant_currently_visible and frames_without_plant < HYSTERESIS:
            return most_common, (0, 255, 0), smoothed
        else:
            plant_currently_visible = False
            frames_without_plant = 0
            return "No Plant Detected", (0, 0, 255), smoothed


def run_motor_spray(severity_level):
    """
    Turns the motor on for the right amount of time based on severity.
    Runs in its own thread so it doesn't block the main loop.
    """
    global spray_in_progress, spray_started_at, spray_lasts_for
    global motor_running, freeze_active, manual_spray_prompt

    if severity_level not in spray_durations:
        return

    duration = spray_durations[severity_level]
    spray_lasts_for = duration
    spray_started_at = time.time()
    motor_running = True

    msg = f"Motor ON for {duration}s — {severity_level.upper()} severity"
    print(msg)
    log_event(msg, "success")

    # Actually wait for the spray to finish
    time.sleep(duration)

    motor_running = False
    spray_in_progress = False
    freeze_active = False
    manual_spray_prompt = ""

    log_event(f"Motor OFF — finished {duration}s spray", "info")
    print("Motor stopped.")


def draw_targeting_box(frame, crop_size=CROP_SIZE):
    """
    Draws a fancy crosshair / targeting box on the center of the frame.
    This visually shows the region the AI is actually analyzing.
    """
    h, w = frame.shape[:2]

    # Calculate where the center crop box starts and ends
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    cx = w // 2
    cy = h // 2

    green = (0, 245, 160)
    white = (255, 255, 255)
    gray = (100, 100, 100)
    bracket_len = 40
    thickness = 3

    # Draw the 4 corner L-shaped brackets
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), green, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), green, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), green, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), green, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), green, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), green, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), green, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), green, thickness)

    # Small crosshair in the center
    cross = 15
    cv2.line(frame, (cx - cross, cy), (cx + cross, cy), white, 2)
    cv2.line(frame, (cx, cy - cross), (cx, cy + cross), white, 2)
    cv2.circle(frame, (cx, cy), 4, green, -1)
    cv2.circle(frame, (cx, cy), 6, white, 1)

    # Dashed border lines for style
    dash_gap = 20
    for i in range(x1 + dash_gap, x2, dash_gap * 2):
        cv2.line(frame, (i, y1), (i + dash_gap, y1), gray, 1)
        cv2.line(frame, (i, y2), (i + dash_gap, y2), gray, 1)
    for i in range(y1 + dash_gap, y2, dash_gap * 2):
        cv2.line(frame, (x1, i), (x1, i + dash_gap), gray, 1)
        cv2.line(frame, (x2, i), (x2, i + dash_gap), gray, 1)

    # Label above the box
    cv2.putText(frame, f"AI ANALYSIS ZONE - {crop_size}x{crop_size}",
                (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)

    return frame, (x1, y1, x2, y2)


def process_frames_loop():
    """
    Background thread that pulls frames from the queue,
    runs both AI models, and pushes the result back.
    """
    global motor_running, freeze_active, spray_in_progress

    while True:
        if not raw_frame_queue:
            time.sleep(0.01)
            continue

        frame = raw_frame_queue.popleft()
        h, w = frame.shape[:2]

        # Get the center crop coordinates
        x1 = (w - CROP_SIZE) // 2
        y1 = (h - CROP_SIZE) // 2
        x2 = x1 + CROP_SIZE
        y2 = y1 + CROP_SIZE

        # Crop and preprocess the center region for the model
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, IMG_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        model_input = np.expand_dims(normalized, axis=0)

        # Step 1 — Is there a plant?
        plant_prob = binary_model.predict(model_input, verbose=0)[0][0]

        if plant_prob > PLANT_DETECTION_THRESHOLD:
            # Step 2 — What severity is the disease?
            severity_preds = severity_model.predict(model_input, verbose=0)
            class_idx = int(np.argmax(severity_preds))
            confidence = float(np.max(severity_preds) * 100)
            label = f"{severity_classes[class_idx]} ({confidence:.1f}%)"
            prediction_history.append((label, confidence))
        else:
            prediction_history.append(("No Plant Detected", 0.0))

        # Get the smoothed/stable result
        stable_label, label_color, stable_conf = get_stable_prediction()

        # Apply the targeting UI overlay
        display = frame.copy()
        display, box_coords = draw_targeting_box(display, CROP_SIZE)
        _, _, _, result_y_top = box_coords
        text_y = result_y_top + 40

        # Write the prediction result on the frame
        cv2.putText(display, f"Plant: {stable_label}",
                    (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        cv2.putText(display, f"Confidence: {stable_conf:.1f}%",
                    (x1, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update the global so the Flask route can read it
        latest_plant["label"] = stable_label
        latest_plant["confidence"] = stable_conf

        # Auto-spray logic — if conditions are right, trigger the motor
        detected_severity = stable_label.split(" ")[0]
        conditions_met = (
            detected_severity in spray_durations
            and latest_moisture < MOISTURE_LIMIT
            and latest_humidity is not None and latest_humidity < HUMIDITY_LIMIT
            and latest_temperature is not None and latest_temperature < TEMP_LIMIT
            and not motor_running
        )
        if conditions_met:
            t = threading.Thread(target=run_motor_spray, args=(detected_severity,), daemon=True)
            t.start()

        # Push the rendered frame to the output queue
        processed_frame_queue.append((stable_label, label_color, stable_conf, display))
        time.sleep(0.01)


def generate_mjpeg_stream():
    """
    Generator function for the MJPEG stream.
    Flask uses this to serve /video_feed as a multipart response.
    """
    while True:
        if processed_frame_queue:
            label, color, conf, frame = processed_frame_queue[-1]
            success, encoded = cv2.imencode('.jpg', frame)
            if success:
                frame_bytes = encoded.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------

@app.route('/process', methods=['POST'])
def handle_moisture_data():
    """
    Pi sends moisture + asks what the motor should do.
    We check conditions and reply with a command.
    """
    global current_motor_command, command_duration, latest_moisture, force_spray_now

    data = request.json
    latest_moisture = data.get('moisture', 0.0)
    print(f"[PI] Moisture received: {latest_moisture}%")

    plant_label = latest_plant.get("label", "").split(" ")[0]

    if force_spray_now:
        # Someone pressed the emergency spray button
        current_motor_command = "RUN"
        command_duration = 3
        force_spray_now = False
        print("Force spray triggered via dashboard.")
        log_event("Manual override — Force spray for 3s", "warning")
    else:
        # Normal auto logic
        all_conditions_ok = (
            plant_label in spray_durations
            and plant_label != "No Plant Detected"
            and latest_moisture < MOISTURE_LIMIT
            and latest_humidity is not None and latest_humidity < HUMIDITY_LIMIT
            and latest_temperature is not None and latest_temperature < TEMP_LIMIT
        )
        if all_conditions_ok:
            current_motor_command = "RUN"
            command_duration = spray_durations[plant_label]
            print(f"Auto trigger: {plant_label} severity, motor ON for {command_duration}s")
            log_event(f"Auto spray: {plant_label}, {command_duration}s", "success")
        else:
            current_motor_command = "STOP"
            command_duration = 0

    return jsonify({"motor_command": current_motor_command, "duration": command_duration})


@app.route('/dht22', methods=['POST'])
def handle_dht22_data():
    """Receives temperature and humidity from the Pi's DHT22 sensor."""
    global latest_temperature, latest_humidity

    data = request.json
    latest_temperature = data.get('temperature')
    latest_humidity = data.get('humidity')
    print(f"[DHT22] Temp: {latest_temperature}°C, Humidity: {latest_humidity}%")

    return jsonify({"status": "received"})


@app.route('/logs')
def get_logs():
    """Returns the activity log as JSON so the dashboard can display it."""
    return jsonify(list(activity_log))


@app.route('/status')
def get_status():
    """Quick status endpoint the dashboard polls every second."""
    return jsonify({
        "plant": latest_plant.get("label", "N/A"),
        "confidence": latest_plant.get("confidence", 0.0),
        "moisture": latest_moisture,
        "temperature": latest_temperature,
        "humidity": latest_humidity,
        "motor": "ON" if motor_running else "OFF"
    })


@app.route('/force_spray', methods=['POST'])
def trigger_force_spray():
    """Dashboard button hit — set the flag so next Pi request triggers a spray."""
    global force_spray_now
    force_spray_now = True
    log_event("Force spray requested from dashboard", "warning")
    return jsonify({"status": "queued"})


@app.route('/video_feed')
def video_feed():
    """Streams the processed webcam frames as MJPEG."""
    return Response(
        generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def dashboard():
    """Serves the main web dashboard."""
    html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant AI Monitor | Intelligent Sprayer System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --green:       #00f5a0;
            --green-dark:  #00c97f;
            --green-glow:  rgba(0, 245, 160, 0.4);
            --red:         #ff3b5c;
            --red-glow:    rgba(255, 59, 92, 0.4);
            --amber:       #ffb020;
            --amber-glow:  rgba(255, 176, 32, 0.4);
            --blue:        #3da9ff;
            --blue-glow:   rgba(61, 169, 255, 0.4);
            --violet:      #8b5cf6;
            --bg:          #060b14;
            --bg-card:     #0f1724;
            --bg-glass:    rgba(15, 23, 36, 0.6);
            --text:        #e6edf6;
            --text-dim:    #9fb3c8;
            --text-muted:  #6b8098;
            --border:      rgba(80, 120, 160, 0.15);
            --radius:      16px;
            --transition:  all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
        }

        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(145deg, #04080f, #060b14);
            background-attachment: fixed;
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Subtle grid background */
        body::before {
            content: "";
            position: fixed; inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(0,245,160,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,245,160,0.04) 1px, transparent 1px);
            background-size: 60px 60px;
            animation: grid-drift 40s linear infinite;
            z-index: -1;
        }
        @keyframes grid-drift { to { transform: translate(-60px, -60px); } }

        /* ---- HEADER ---- */
        .top-bar {
            background: rgba(15, 23, 36, 0.75);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(120, 170, 220, 0.2);
            padding: 1.1rem 0;
            position: relative;
            overflow: hidden;
        }
        .top-bar::after {
            content: "";
            position: absolute; bottom: 0; left: -60%;
            width: 50%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--green), transparent);
            animation: scan 6s linear infinite;
        }
        @keyframes scan { to { left: 110%; } }

        .site-title {
            font-size: 1.55rem; font-weight: 800;
            letter-spacing: 1px; text-transform: uppercase;
            background: linear-gradient(135deg, #fff, var(--green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .system-badge {
            background: linear-gradient(135deg, var(--green), var(--green-dark));
            color: #03110a;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.7rem; font-weight: 700;
            letter-spacing: 1.2px; text-transform: uppercase;
            animation: badge-glow 2s ease-in-out infinite alternate;
        }
        @keyframes badge-glow {
            from { box-shadow: 0 0 10px var(--green-glow); }
            to   { box-shadow: 0 0 22px var(--green-glow); }
        }

        .online-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 10px var(--green);
            animation: dot-pulse 2.5s ease-in-out infinite;
        }
        @keyframes dot-pulse {
            0%, 100% { transform: scale(1); }
            50%       { transform: scale(1.2); box-shadow: 0 0 18px var(--green); }
        }

        /* ---- KPI CARDS ---- */
        .kpi-card {
            background: linear-gradient(145deg, rgba(15,23,36,0.8), var(--bg-card));
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            position: relative; overflow: hidden;
            backdrop-filter: blur(14px);
            transition: var(--transition);
            cursor: pointer;
            height: 100%;
        }
        .kpi-card::before {
            content: "";
            position: absolute; top: 0; left: 0;
            width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--green), transparent);
            transform: scaleX(0); transform-origin: left;
            transition: transform 0.4s ease;
        }
        .kpi-card:hover { transform: translateY(-7px); border-color: rgba(0,245,160,0.35); }
        .kpi-card:hover::before { transform: scaleX(1); }

        .kpi-icon {
            width: 48px; height: 48px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(0,245,160,0.15);
            background: rgba(0,245,160,0.07);
            transition: var(--transition);
        }
        .kpi-card:hover .kpi-icon { transform: scale(1.1) rotate(5deg); }

        .kpi-icon.green  { color: var(--green);  border-color: rgba(0,245,160,0.3);  background: rgba(0,245,160,0.1);  }
        .kpi-icon.red    { color: var(--red);    border-color: rgba(255,59,92,0.3);  background: rgba(255,59,92,0.1);  }
        .kpi-icon.blue   { color: var(--blue);   border-color: rgba(61,169,255,0.3); background: rgba(61,169,255,0.1); }
        .kpi-icon.amber  { color: var(--amber);  border-color: rgba(255,176,32,0.3); background: rgba(255,176,32,0.1); }
        .kpi-icon.cyan   { color: #22d3ee;       border-color: rgba(34,211,238,0.3); background: rgba(34,211,238,0.1); }
        .kpi-icon.violet { color: var(--violet); border-color: rgba(139,92,246,0.3); background: rgba(139,92,246,0.1); }

        .kpi-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.3px; color: var(--text-dim); margin-bottom: 0.5rem; }
        .kpi-value { font-size: 1.85rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.3rem; transition: transform 0.3s ease; }
        .kpi-card:hover .kpi-value { transform: scale(1.04); }
        .kpi-value.green  { color: var(--green); }
        .kpi-value.blue   { color: var(--blue); }
        .kpi-value.amber  { color: var(--amber); }
        .kpi-value.red    { color: var(--red); }
        .kpi-subtext { font-size: 0.82rem; color: var(--text-dim); }

        /* ---- FORCE SPRAY SECTION ---- */
        .force-spray-box {
            background: linear-gradient(145deg, rgba(255,59,92,0.08), rgba(255,59,92,0.02));
            border: 1px solid rgba(255,59,92,0.25);
            border-radius: var(--radius);
            padding: 1.4rem;
            margin-bottom: 1.5rem;
            animation: spray-box-pulse 3s ease-in-out infinite;
        }
        @keyframes spray-box-pulse {
            0%, 100% { border-color: rgba(255,59,92,0.25); }
            50%       { border-color: rgba(255,59,92,0.5); box-shadow: 0 0 25px rgba(255,59,92,0.15); }
        }

        .force-btn {
            width: 100%; padding: 0.9rem 1.4rem;
            border: none; border-radius: 12px;
            background: linear-gradient(135deg, var(--red), #d91e3f);
            color: #fff; font-weight: 700; font-size: 0.95rem;
            text-transform: uppercase; letter-spacing: 1px;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center; gap: 0.7rem;
            box-shadow: 0 0 15px var(--red-glow);
            transition: var(--transition);
            animation: btn-pulse 2s ease-in-out infinite;
        }
        @keyframes btn-pulse {
            0%, 100% { box-shadow: 0 0 15px var(--red-glow); }
            50%       { box-shadow: 0 0 28px var(--red-glow); }
        }
        .force-btn:hover:not(:disabled) { transform: translateY(-3px) scale(1.02); }
        .force-btn:disabled { opacity: 0.55; cursor: not-allowed; animation: none; }

        /* ---- CAMERA & OPERATION PANEL ---- */
        .panel {
            background: linear-gradient(180deg, rgba(255,255,255,0.015), transparent), var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
            transition: var(--transition);
        }
        .panel:hover { transform: translateY(-4px); border-color: rgba(255,255,255,0.1); }

        .panel-header {
            background: linear-gradient(135deg, rgba(0,245,160,0.08), transparent);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: 0.75rem;
            position: relative;
        }
        .panel-header::before {
            content: "";
            position: absolute; left: 0; top: 0; bottom: 0;
            width: 4px;
            background: var(--green);
            box-shadow: 0 0 15px var(--green-glow);
        }
        .panel-header h5 { margin: 0; font-weight: 600; font-size: 1rem; }

        /* ---- LIVE FEED ---- */
        .camera-box {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            aspect-ratio: 16/9;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .camera-box img { width: 100%; height: 100%; object-fit: cover; min-height: 280px; }

        .live-badge {
            position: absolute; top: 10px; left: 10px;
            background: linear-gradient(135deg, var(--red), rgba(255,59,92,0.85));
            color: #fff; padding: 0.28rem 0.75rem;
            border-radius: 6px; font-size: 0.68rem; font-weight: 700;
            letter-spacing: 1.1px; z-index: 10;
            animation: live-blink 1.5s ease-in-out infinite;
        }
        @keyframes live-blink {
            0%, 100% { opacity: 1; } 50% { opacity: 0.75; }
        }

        .detection-bar {
            position: absolute; bottom: 10px; left: 10px; right: 10px;
            background: rgba(6, 11, 20, 0.9);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0,245,160,0.25);
            border-radius: 10px;
            padding: 0.7rem 1rem;
            display: flex; align-items: center; gap: 0.7rem;
            z-index: 10;
        }
        .detection-label { font-weight: 700; font-size: 0.85rem; text-transform: uppercase; }
        .detection-label.healthy { color: var(--green); }
        .detection-label.infected { color: var(--red); }
        .detection-label.warning  { color: var(--amber); }
        .detection-conf { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-dim); }

        /* ---- SPRAY DURATION PANEL ---- */
        .duration-panel {
            background: linear-gradient(145deg, rgba(15,23,36,0.8), var(--bg-card));
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.4rem;
            height: 100%;
            backdrop-filter: blur(14px);
        }
        .duration-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.7rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.05);
            transition: transform 0.25s ease;
        }
        .duration-row:hover { transform: translateX(7px); }
        .duration-row.low    { border-left: 4px solid var(--amber); }
        .duration-row.medium { border-left: 4px solid #f97316; }
        .duration-row.high   { border-left: 4px solid var(--red); }

        .sev-badge {
            padding: 0.3rem 0.7rem; border-radius: 6px;
            font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        }
        .sev-badge.low    { background: rgba(255,176,32,0.2); color: var(--amber); }
        .sev-badge.medium { background: rgba(249,115,22,0.2); color: #f97316; }
        .sev-badge.high   { background: rgba(255,59,92,0.2);  color: var(--red); }

        .sev-time { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; }

        /* ---- CHART ---- */
        .chart-box {
            background: rgba(15, 23, 36, 0.9);
            border: 1px solid var(--green-glow);
            border-radius: 20px;
            padding: 1.5rem;
            height: 360px;
            box-shadow: 0 4px 20px rgba(0,245,160,0.15);
            backdrop-filter: blur(12px);
        }

        /* ---- ACTIVITY LOG ---- */
        .log-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }
        .log-header {
            background: linear-gradient(135deg, rgba(61,169,255,0.12), rgba(0,245,160,0.04));
            padding: 0.9rem 1.2rem;
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: 0.7rem;
        }
        .log-body { max-height: 320px; overflow-y: auto; padding: 0.5rem; }
        .log-entry {
            display: flex; align-items: flex-start; gap: 0.7rem;
            padding: 0.7rem 0.9rem;
            border-radius: 10px;
            margin-bottom: 0.45rem;
            border-left: 3px solid transparent;
            transition: all 0.25s ease;
            animation: slide-in 0.3s ease-out;
        }
        @keyframes slide-in { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: none; } }
        .log-entry:hover { transform: translateX(4px); }
        .log-entry.success { border-color: var(--green);  background: rgba(0,245,160,0.07); }
        .log-entry.warning { border-color: var(--amber);  background: rgba(255,176,32,0.07); }
        .log-entry.error   { border-color: var(--red);    background: rgba(255,59,92,0.07); }
        .log-entry.info    { border-color: var(--blue);   background: rgba(61,169,255,0.07); }

        .log-icon {
            width: 34px; height: 34px; border-radius: 9px;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; font-size: 0.85rem;
        }
        .log-icon.success { background: rgba(0,245,160,0.15); color: var(--green); }
        .log-icon.warning { background: rgba(255,176,32,0.15); color: var(--amber); }
        .log-icon.error   { background: rgba(255,59,92,0.15);  color: var(--red); }
        .log-icon.info    { background: rgba(61,169,255,0.15); color: var(--blue); }

        .log-text { font-size: 0.85rem; margin-bottom: 0.2rem; }
        .log-time { font-size: 0.72rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

        /* ---- TOAST ---- */
        .toast-area { position: fixed; bottom: 18px; right: 18px; z-index: 9999; display: flex; flex-direction: column; gap: 0.65rem; }
        .toast-msg {
            background: rgba(15,23,36,0.98);
            border: 1px solid var(--green-glow);
            border-radius: 14px; padding: 0.9rem 1.1rem;
            display: flex; align-items: center; gap: 0.7rem;
            box-shadow: 0 8px 30px rgba(0,245,160,0.2);
            min-width: 260px; font-weight: 600;
            backdrop-filter: blur(10px);
            animation: toast-in 0.35s ease-out;
        }
        @keyframes toast-in { from { transform: translateX(100%); opacity: 0; } to { transform: none; opacity: 1; } }
        .toast-msg.success { border-left: 4px solid var(--green); }
        .toast-msg.error   { border-left: 4px solid var(--red); }
        .toast-msg.warning { border-left: 4px solid var(--amber); }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); border-radius: 8px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--green), var(--green-dark)); border-radius: 8px; }
    </style>
</head>
<body>

    <!-- Top header bar -->
    <header class="top-bar">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <div class="d-flex align-items-center gap-3">
                        <i class="fas fa-leaf text-success fs-3"></i>
                        <div>
                            <div class="site-title">Plant AI Monitor</div>
                            <span class="system-badge">Intelligent Sprayer System Pro</span>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 text-md-end mt-2 mt-md-0">
                    <div class="d-flex align-items-center justify-content-md-end gap-2">
                        <div class="online-dot" id="onlineDot"></div>
                        <span id="statusLabel" style="font-size:0.85rem; color:var(--text-dim);">System Online</span>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="container py-4">

        <!-- KPI cards row -->
        <div class="row g-3 mb-4">
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon green"><i class="fas fa-seedling"></i></div>
                    <div class="kpi-label">Plant Status</div>
                    <div class="kpi-value green" id="plantStatus">--</div>
                    <div class="kpi-subtext" id="plantSub">Waiting...</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon blue"><i class="fas fa-brain"></i></div>
                    <div class="kpi-label">AI Confidence</div>
                    <div class="kpi-value blue" id="aiConf">--%</div>
                    <div class="kpi-subtext">Neural Network</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon blue"><i class="fas fa-tint"></i></div>
                    <div class="kpi-label">Soil Moisture</div>
                    <div class="kpi-value blue" id="moistureVal">--%</div>
                    <div class="kpi-subtext" id="moistureSub">Waiting...</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon amber"><i class="fas fa-thermometer-half"></i></div>
                    <div class="kpi-label">Temperature</div>
                    <div class="kpi-value amber" id="tempVal">--°C</div>
                    <div class="kpi-subtext">Ambient</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon cyan"><i class="fas fa-water"></i></div>
                    <div class="kpi-label">Humidity</div>
                    <div class="kpi-value blue" id="humidVal">--%</div>
                    <div class="kpi-subtext">Air Moisture</div>
                </div>
            </div>
            <div class="col-6 col-md-4 col-lg-2">
                <div class="kpi-card">
                    <div class="kpi-icon violet"><i class="fas fa-cog" id="motorIcon"></i></div>
                    <div class="kpi-label">Motor Status</div>
                    <div class="kpi-value" id="motorStatus">IDLE</div>
                    <div class="kpi-subtext" id="motorSub">Ready</div>
                </div>
            </div>
        </div>

        <!-- Force spray emergency button -->
        <div class="force-spray-box">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="d-flex align-items-center gap-3">
                        <div class="kpi-icon red" style="margin-bottom:0;"><i class="fas fa-exclamation-triangle"></i></div>
                        <div>
                            <h5 style="color:var(--red); margin-bottom:0.3rem;">
                                <i class="fas fa-spray-can me-2"></i>Manual Override
                            </h5>
                            <p class="mb-0 text-muted" style="font-size:0.85rem;">
                                Emergency spray — triggers 3-second spray ignoring all conditions
                            </p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mt-3 mt-md-0">
                    <button class="force-btn" id="forceBtn" onclick="triggerForceSpray()">
                        <i class="fas fa-exclamation-triangle"></i> FORCE SPRAY
                    </button>
                </div>
            </div>
        </div>

        <!-- Camera feed + spray duration side by side -->
        <div class="row g-4 mb-4">
            <div class="col-lg-8">
                <div class="panel">
                    <div class="panel-header">
                        <i class="fas fa-video" style="color:var(--green);"></i>
                        <h5>Live Camera Feed (HD 720p)</h5>
                        <span class="badge bg-danger ms-auto">LIVE</span>
                    </div>
                    <div class="p-3">
                        <div class="camera-box" id="cameraBox">
                            <img id="videoFeed" src="/video_feed" alt="Live Camera Feed">
                            <span class="live-badge">LIVE</span>
                            <div class="detection-bar" id="detectionBar">
                                <div class="detection-label" id="detectionLabel">
                                    <i class="fas fa-search me-1"></i> Analyzing...
                                </div>
                                <span class="detection-conf" id="detectionConf">--</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="duration-panel">
                    <div class="d-flex align-items-center gap-2 mb-3">
                        <i class="fas fa-clock" style="color:var(--green);"></i>
                        <h6 class="mb-0" style="font-weight:700; text-transform:uppercase; letter-spacing:1px;">
                            Auto-Spray Durations
                        </h6>
                    </div>
                    <p class="text-muted mb-3" style="font-size:0.82rem;">
                        Motor run time based on detected disease severity
                    </p>
                    <div class="duration-row low">
                        <div class="d-flex align-items-center gap-2">
                            <span class="sev-badge low">LOW</span>
                            <span class="text-muted" style="font-size:0.8rem;">Mild infection</span>
                        </div>
                        <span class="sev-time" style="color:var(--amber);">2s</span>
                    </div>
                    <div class="duration-row medium">
                        <div class="d-flex align-items-center gap-2">
                            <span class="sev-badge medium">MED</span>
                            <span class="text-muted" style="font-size:0.8rem;">Moderate infection</span>
                        </div>
                        <span class="sev-time" style="color:#f97316;">3s</span>
                    </div>
                    <div class="duration-row high">
                        <div class="d-flex align-items-center gap-2">
                            <span class="sev-badge high">HIGH</span>
                            <span class="text-muted" style="font-size:0.8rem;">Severe infection</span>
                        </div>
                        <span class="sev-time" style="color:var(--red);">5s</span>
                    </div>
                    <div class="mt-3 p-2 rounded" style="background:rgba(0,245,160,0.04); border:1px solid rgba(0,245,160,0.1);">
                        <small style="color:var(--text-muted); font-size:0.75rem;">
                            <i class="fas fa-info-circle me-1" style="color:var(--green);"></i>
                            System adjusts spray time automatically based on AI confidence
                        </small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Moisture history chart -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="chart-box">
                    <div class="d-flex align-items-center justify-content-between mb-3">
                        <h5 class="mb-0" style="font-weight:700;">
                            <i class="fas fa-chart-area me-2" style="color:var(--green);"></i>
                            Soil Moisture History
                        </h5>
                        <span class="text-muted" style="font-size:0.82rem;">Last 30 readings</span>
                    </div>
                    <canvas id="moistureChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Activity log -->
        <div class="row">
            <div class="col-12">
                <div class="log-box">
                    <div class="log-header">
                        <i class="fas fa-terminal" style="color:var(--blue);"></i>
                        <h6 class="mb-0" style="font-weight:600;">System Activity Log</h6>
                        <span class="badge bg-secondary ms-auto" id="logCount">0</span>
                    </div>
                    <div class="log-body" id="logBody">
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-circle-notch fa-spin mb-2"></i>
                            <p class="mb-0" style="font-size:0.85rem;">Loading logs...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <div class="toast-area" id="toastArea"></div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // ---- Moisture Chart Setup ----
        const chartCtx = document.getElementById('moistureChart').getContext('2d');
        let moistureData = Array(30).fill(50);

        const moistureChart = new Chart(chartCtx, {
            type: 'line',
            data: {
                labels: moistureData.map((_, i) => `T-${30 - i}`),
                datasets: [{
                    label: 'Soil Moisture (%)',
                    data: moistureData,
                    borderColor: '#00f5a0',
                    backgroundColor: 'rgba(0, 245, 160, 0.08)',
                    borderWidth: 2,
                    pointRadius: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#9fb3c8' } } },
                scales: {
                    x: { ticks: { color: '#6b8098' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: {
                        min: 0, max: 100,
                        ticks: { color: '#6b8098' },
                        grid: { color: 'rgba(255,255,255,0.04)' }
                    }
                }
            }
        });

        // ---- Push a toast notification ----
        function showToast(message, type = 'success') {
            const area = document.getElementById('toastArea');
            const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle' };
            const colors = { success: '#00f5a0', error: '#ff3b5c', warning: '#ffb020' };

            const toast = document.createElement('div');
            toast.className = `toast-msg ${type}`;
            toast.innerHTML = `
                <i class="fas ${icons[type] || 'fa-info-circle'}" style="color:${colors[type]};"></i>
                <span>${message}</span>`;
            area.appendChild(toast);

            setTimeout(() => toast.remove(), 3500);
        }

        // ---- Force spray button ----
        async function triggerForceSpray() {
            const btn = document.getElementById('forceBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Spraying...';

            try {
                const res = await fetch('/force_spray', { method: 'POST' });
                const data = await res.json();
                showToast('Force spray activated — 3s spray queued!', 'warning');
            } catch (err) {
                showToast('Failed to trigger spray', 'error');
            }

            setTimeout(() => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> FORCE SPRAY';
            }, 4000);
        }

        // ---- Poll status and update all KPI cards ----
        async function refreshStatus() {
            try {
                const res = await fetch('/status');
                const d = await res.json();

                // Plant status card
                const plantLabel = (d.plant || '').split(' ')[0].toLowerCase();
                document.getElementById('plantStatus').textContent = plantLabel || '--';
                document.getElementById('plantSub').textContent = `${(d.confidence || 0).toFixed(1)}% confidence`;

                // AI confidence
                document.getElementById('aiConf').textContent = `${(d.confidence || 0).toFixed(1)}%`;

                // Moisture
                const moisture = d.moisture ?? 0;
                document.getElementById('moistureVal').textContent = `${moisture.toFixed(1)}%`;
                document.getElementById('moistureSub').textContent = moisture < 40 ? 'Low — needs water' : 'OK';

                // Moisture chart update
                moistureData.push(moisture);
                if (moistureData.length > 30) moistureData.shift();
                moistureChart.data.datasets[0].data = [...moistureData];
                moistureChart.update('none');

                // Temperature
                const temp = d.temperature;
                document.getElementById('tempVal').textContent = temp !== null ? `${temp}°C` : '--°C';

                // Humidity
                const hum = d.humidity;
                document.getElementById('humidVal').textContent = hum !== null ? `${hum}%` : '--%';

                // Motor status
                const motorEl = document.getElementById('motorStatus');
                const motorSub = document.getElementById('motorSub');
                const motorIcon = document.getElementById('motorIcon');
                if (d.motor === 'ON') {
                    motorEl.textContent = 'ACTIVE';
                    motorEl.style.color = '#00f5a0';
                    motorSub.textContent = 'Spraying...';
                    motorIcon.style.animation = 'spin 1s linear infinite';
                } else {
                    motorEl.textContent = 'IDLE';
                    motorEl.style.color = '';
                    motorSub.textContent = 'Ready';
                    motorIcon.style.animation = '';
                }

                // Detection bar below video
                const detLabel = document.getElementById('detectionLabel');
                const detConf = document.getElementById('detectionConf');
                detLabel.textContent = d.plant || 'Analyzing...';
                detConf.textContent = `${(d.confidence || 0).toFixed(1)}%`;

            } catch (err) {
                // server might be offline, just skip
            }
        }

        // ---- Poll activity log ----
        async function refreshLogs() {
            try {
                const res = await fetch('/logs');
                const logs = await res.json();

                const body = document.getElementById('logBody');
                const count = document.getElementById('logCount');
                count.textContent = logs.length;

                if (logs.length === 0) {
                    body.innerHTML = '<div class="text-center text-muted py-3" style="font-size:0.85rem;">No activity yet.</div>';
                    return;
                }

                const typeIcons = {
                    success: 'fa-check',
                    warning: 'fa-exclamation',
                    error:   'fa-times',
                    info:    'fa-info'
                };

                body.innerHTML = [...logs].reverse().map(entry => `
                    <div class="log-entry ${entry.type}">
                        <div class="log-icon ${entry.type}">
                            <i class="fas ${typeIcons[entry.type] || 'fa-info'}"></i>
                        </div>
                        <div>
                            <div class="log-text">${entry.message}</div>
                            <div class="log-time">${entry.time}</div>
                        </div>
                    </div>
                `).join('');

            } catch (err) {
                // just skip if server is down
            }
        }

        // ---- Spin animation for motor icon ----
        const motorStyle = document.createElement('style');
        motorStyle.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
        document.head.appendChild(motorStyle);

        // ---- Start polling ----
        refreshStatus();
        refreshLogs();
        setInterval(refreshStatus, 1500);
        setInterval(refreshLogs, 3000);
    </script>

</body>
</html>
"""
    return render_template_string(html_page)


# -------------------------------------------------------
# Entry point — start everything and run the server
# -------------------------------------------------------
if __name__ == '__main__':
    # Open the webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam opened successfully.")

    # Start the AI processing thread
    ai_thread = threading.Thread(target=process_frames_loop, daemon=True)
    ai_thread.start()

    # Start a thread to continuously grab frames from the webcam
    def capture_loop():
        while True:
            ret, frame = cap.read()
            if ret:
                raw_frame_queue.append(frame)
            time.sleep(0.03)  # ~30fps

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    print("Starting Flask server on http://0.0.0.0:5000")
    log_event("System started", "info")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
