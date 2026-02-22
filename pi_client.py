import io
import time
import logging
import threading
import requests
from flask import Flask, Response
from gpiozero import OutputDevice
import picamera
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


# -------------------------------------------------------
# Config — update these before running
# -------------------------------------------------------
LAPTOP_IP = "10.137.85.201"   # Change this to your laptop's IP address
SERVER_URL = f"http://{LAPTOP_IP}:5000/process"

RELAY_GPIO_PIN = 18  # GPIO 18 controls the water pump relay (physical pin 12)


# -------------------------------------------------------
# Hardware setup
# -------------------------------------------------------

# Motor/pump connected through a relay on GPIO 18
# active_high=False because relay triggers when signal goes LOW
motor = OutputDevice(RELAY_GPIO_PIN, active_high=False, initial_value=False)

# ADS1115 ADC for reading the analog soil moisture sensor over I2C
i2c_bus = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c_bus)
moisture_channel = AnalogIn(ads, ADS.P0)  # Moisture sensor plugged into channel 0


# -------------------------------------------------------
# Flask app — used to stream the Pi camera video feed
# -------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Holds the latest moisture reading so we can share it between threads
current_moisture = 0.0


def read_soil_moisture():
    """
    Reads the raw ADC value and converts it into a percentage.
    The sensor gives higher voltage when dry and lower when wet,
    so we invert it: moisture% = 100 - (raw / max * 100)
    """
    global current_moisture

    raw_value = moisture_channel.value
    # 26000 is roughly the max dry reading from our sensor — calibrate if needed
    moisture_pct = round(100 - (raw_value / 26000 * 100), 1)

    # Clamp between 0 and 100 just in case sensor reads weird
    current_moisture = max(0.0, min(100.0, moisture_pct))


def camera_stream():
    """
    Opens the Pi camera and yields JPEG frames in MJPEG format.
    This runs continuously as a Flask streaming response.
    """
    try:
        with picamera.PiCamera(resolution='320x240', framerate=15) as cam:
            logging.info("Pi Camera opened successfully.")

            cam.exposure_mode = 'auto'
            cam.iso = 200
            cam.awb_mode = 'auto'

            # Give the camera a moment to warm up and auto-adjust
            time.sleep(3)

            frame_buffer = io.BytesIO()
            for _ in cam.capture_continuous(frame_buffer, 'jpeg', use_video_port=True, quality=50):
                frame_buffer.seek(0)
                frame_bytes = frame_buffer.read()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Reset the buffer for next frame
                frame_buffer.seek(0)
                frame_buffer.truncate()
                time.sleep(0.02)

    except Exception as e:
        logging.error(f"Camera error: {e}")


@app.route('/video_feed')
def video_feed():
    """Flask route that returns the MJPEG camera stream."""
    return Response(
        camera_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def sensor_and_motor_loop():
    """
    Main loop that runs in a background thread:
    1. Reads soil moisture from ADS1115
    2. Sends it to the laptop server
    3. Receives motor command
    4. Controls the pump accordingly
    """
    while True:
        read_soil_moisture()

        payload = {"moisture": current_moisture}

        try:
            response = requests.post(SERVER_URL, json=payload, timeout=2)
            result = response.json()

            command = result.get("motor_command")
            run_for = result.get("duration", 0)

            logging.info(f"Moisture: {current_moisture}% | Command: {command} | Duration: {run_for}s")

            if command == "RUN" and run_for > 0:
                motor.on()
                logging.info(f"Motor ON for {run_for}s")
                time.sleep(run_for)
                motor.off()
                logging.info("Motor OFF")
            else:
                # Make sure motor is off if not supposed to run
                motor.off()

        except Exception as e:
            logging.error(f"Failed to reach server: {e}")
            motor.off()  # Safety — always turn off motor if connection fails

        time.sleep(1)  # Wait 1 second before next reading


if __name__ == '__main__':
    # Start the moisture-reading + motor-control loop in a background thread
    sensor_thread = threading.Thread(target=sensor_and_motor_loop, daemon=True)
    sensor_thread.start()

    # Start the Flask server to stream camera footage
    logging.info(f"Starting Pi client — sending data to {SERVER_URL}")
    app.run(host='0.0.0.0', port=5000, debug=False)
