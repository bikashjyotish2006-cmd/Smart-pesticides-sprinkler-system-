# 🌿 AI Plant Disease Detection & Smart Irrigation System

An end-to-end intelligent plant monitoring and automated irrigation system that uses deep learning to detect plant diseases in real time via webcam, and triggers a water pump based on disease severity and environmental sensor data.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Technologies & Frameworks](#technologies--frameworks)
- [Python Libraries](#python-libraries)
- [Frontend Technologies](#frontend-technologies)
- [Hardware Components](#hardware-components)
- [System Architecture](#system-architecture)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)

---

## 🚀 Project Overview

This system runs on two devices:
- **Laptop / PC (Server)** — Runs `main.py`: hosts a Flask server, performs real-time AI inference on the live camera feed, and presents a web dashboard.
- **Raspberry Pi (Client)** — Runs `pi_client.py` and `dht22_client.py`: reads soil moisture and DHT22 (temp/humidity) sensors, streams the camera feed, and controls the water pump motor via GPIO relay.

---

## 🧠 Technologies & Frameworks

### AI / Deep Learning
| Technology | Version | Purpose |
|---|---|---|
| **TensorFlow** | 2.20.0 | Deep learning backend |
| **Keras** | 3.12.1 | Model loading & inference (`load_model`) |
| **MobileNetV2** | — | CNN architecture used for both trained models |

### Computer Vision
| Technology | Version | Purpose |
|---|---|---|
| **OpenCV (`cv2`)** | 4.13.0 | Real-time webcam capture, frame processing, ROI extraction, HUD overlay rendering |
| **NumPy** | — | Array operations, image preprocessing, normalization |

### Web Backend
| Technology | Version | Purpose |
|---|---|---|
| **Flask** | 3.1.2 | REST API server, MJPEG video streaming, dashboard route, `/process` and `/dht22` endpoints |
| **Python `threading`** | — | Multi-threaded frame processing and sensor polling |
| **Python `collections.deque`** | — | Circular buffers for frame queues and temporal smoothing |

### Client-Side (Raspberry Pi)
| Technology | Version | Purpose |
|---|---|---|
| **`picamera`** | — | Raspberry Pi Camera Module interface for MJPEG streaming |
| **`gpiozero`** | — | GPIO control for water pump relay (OutputDevice) |
| **`adafruit-ads1x15`** | — | I2C driver for ADS1115 16-bit ADC (reads analog soil moisture sensor) |
| **`adafruit-circuitpython-busdevice`** | — | I2C bus communication (`board`, `busio`) |
| **`Adafruit_DHT`** | — | DHT22 temperature and humidity sensor reading |
| **`requests`** | — | HTTP POST sensor data to the laptop Flask server |

---

## 🌐 Frontend Technologies

The web dashboard is served inline from `main.py` as a `render_template_string` response.

| Technology | Purpose |
|---|---|
| **HTML5** | Dashboard structure and semantic layout |
| **CSS3 (Vanilla)** | Custom dark-mode UI, glassmorphism cards, keyframe animations, CSS variables |
| **JavaScript (Vanilla ES6)** | Live data polling (`fetch`), DOM updates, Chart.js integration, overlay modals, toast notifications |
| **Bootstrap 5.3.2** | Responsive grid layout and utility classes (via CDN) |
| **Chart.js** | Real-time soil moisture history line chart (via CDN) |
| **Font Awesome 6.4.2** | Icon library for UI indicators (via CDN) |
| **Google Fonts** | `Inter` (UI font) and `JetBrains Mono` (monospace data display) |

---

## ⚙️ Python Libraries Summary

```
Python        3.10.11
TensorFlow    2.20.0
Keras         3.12.1
OpenCV        4.13.0
Flask         3.1.2
NumPy         (latest compatible)
requests      (latest compatible)
picamera      (Raspberry Pi only)
gpiozero      (Raspberry Pi only)
adafruit-ads1x15   (Raspberry Pi only)
Adafruit-DHT       (Raspberry Pi only)
```

---

## 🔧 Hardware Components

| Component | Details | Role |
|---|---|---|
| **Raspberry Pi** (3B+ / 4) | ARM-based Linux SBC | Edge client: sensor reading, camera streaming, motor control |
| **Raspberry Pi Camera Module** | V1 / V2 / HQ | Captures live MJPEG video at 320×240 @ 15fps |
| **Laptop / Desktop PC** | Any OS (Windows/macOS/Linux) with GPU optional | Runs Flask server + AI inference |
| **DHT22 Sensor** | Digital temperature & humidity sensor | Reads ambient temperature (°C) and humidity (%) via GPIO 22 |
| **Capacitive Soil Moisture Sensor** | Analog (0–3.3V output) | Reads soil moisture percentage via ADS1115 ADC, Channel P0 |
| **ADS1115 ADC Module** | 16-bit, I2C, 4-channel | Converts analog moisture sensor output to digital via I2C (SCL/SDA) |
| **5V Relay Module** | Single channel | Electrically isolates low-voltage GPIO from the water pump circuit |
| **Water Pump / Motor** | DC submersible or peristaltic | Activated via relay on GPIO 18 (physical pin 12) for timed irrigation |
| **GPU** (optional) | NVIDIA CUDA-compatible | Accelerates TensorFlow inference on the laptop server |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       LAPTOP / PC (Server)                      │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Webcam Feed │───▶│  OpenCV +    │───▶│   MobileNetV2    │   │
│  │  HD 720p    │    │  ROI Extract │    │  Binary + Severity│   │
│  └─────────────┘    └──────────────┘    └──────────┬───────┘   │
│                                                     │           │
│  ┌─────────────────────────────────────────────────▼───────┐   │
│  │              Flask Web Server (port 5000)               │   │
│  │  /          → Web Dashboard (HTML/CSS/JS)               │   │
│  │  /video_feed → MJPEG Stream                             │   │
│  │  /process   → Receive moisture, send motor command      │   │
│  │  /dht22     → Receive temp/humidity                     │   │
│  │  /logs      → Return system activity log                │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ HTTP (LAN)
┌──────────────────────────────────▼──────────────────────────────┐
│                      RASPBERRY PI (Client)                       │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Pi Camera   │  │ ADS1115 ADC │  │ DHT22 Sensor            │  │
│  │ (picamera)  │  │ I2C / GPIO  │  │ GPIO 22                 │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         │                │                       │               │
│  ┌──────▼──────┐  ┌──────▼──────────────────────▼────────────┐  │
│  │ Video Feed  │  │    pi_client.py / dht22_client.py        │  │
│  │ /video_feed │  │    POST moisture → /process              │  │
│  │ (MJPEG)     │  │    POST temp/hum → /dht22                │  │
│  └─────────────┘  └───────────────────────┬──────────────────┘  │
│                                           │ GPIO 18              │
│                                   ┌───────▼─────┐               │
│                                   │  Relay +    │               │
│                                   │ Water Pump  │               │
│                                   └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
Git_upload/
│
├── main.py                                          # Laptop server: Flask API + AI inference + Web Dashboard
├── pi_client.py                                     # Raspberry Pi: moisture sensor + camera + motor control
├── dht22_client.py                                  # Raspberry Pi: DHT22 temperature & humidity sender
│
├── plant_vs_nonplant_mobilenetv2_final.h5           # Trained binary AI model (Plant vs Non-Plant)
├── suraj_chand_severity_mobilenetv2_optimized_      # Trained severity AI model (Healthy / Low / Medium / High)
│   final_ooooop.h5
│
└── README.md                                        # This file
```

---

## 🔄 How It Works

1. **Camera Feed** — A webcam (or Pi Camera) streams live HD video. A 300×300 center ROI is extracted for AI analysis.
2. **Binary Classification** — The `plant_vs_nonplant` MobileNetV2 model checks if a plant is present (threshold: 99%).
3. **Severity Classification** — If a plant is detected, the `severity` MobileNetV2 model classifies it as `Healthy`, `Low`, `Medium`, or `High` infection.
4. **Temporal Smoothing** — A rolling deque of 20 frames with exponential smoothing prevents flickering predictions.
5. **Sensor Data** — The Raspberry Pi sends soil moisture, temperature, and humidity every second via HTTP POST.
6. **Auto Irrigation Logic** — If disease severity is detected AND soil moisture < 40% AND humidity < 70% AND temperature < 30°C, the server commands the Pi to run the pump for 2–5 seconds based on severity.
7. **Web Dashboard** — A real-time dashboard shows live camera feed, KPI cards, moisture history chart, and system activity logs.

---

## 👥 Team

> Developed by **Suraj & Chand** — AI Plant Disease Detection & Smart Irrigation System
