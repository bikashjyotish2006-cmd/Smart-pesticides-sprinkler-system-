 AI Plant Disease Detection & Smart Irrigation

This folder contains the clean, human-written version of the project code.

## Files

| File | Runs On | What it does |
|---|---|---|
| `main.py` | Laptop / PC | Flask server + AI inference + Web dashboard |
| `pi_client.py` | Raspberry Pi | Reads soil moisture, controls motor, streams camera |
| `dht22_client.py` | Raspberry Pi | Reads DHT22 sensor and sends temp/humidity to server |

## Before Running

1. Update `LAPTOP_IP` in both `pi_client.py` and `dht22_client.py` with your laptop's local IP.
2. Update the model file paths in `main.py` if your `.h5` files are in a different location.
3. Make sure all dependencies from `requirements.txt` (in parent folder) are installed.

## Run Order

1. Start `main.py` on the laptop first.
2. Start `pi_client.py` on the Pi.
3. Start `dht22_client.py` on the Pi (separate terminal).
4. Open `http://<laptop-ip>:5000` in a browser for the dashboard.
