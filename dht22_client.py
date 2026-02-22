import time
import requests
import Adafruit_DHT as dht_sensor


# -------------------------------------------------------
# Config
# -------------------------------------------------------
LAPTOP_IP = "10.137.85.201"   # Change to your laptop's IP
SERVER_URL = f"http://{LAPTOP_IP}:5000/dht22"

DHT_SENSOR_TYPE = dht_sensor.DHT22
DHT_GPIO_PIN = 22  # GPIO 22, physical pin 15


# -------------------------------------------------------
# Main loop — reads temp & humidity and sends to server
# -------------------------------------------------------
print(f"DHT22 sender started — sending data to {SERVER_URL}")

while True:
    # Read from the DHT22 sensor
    # dht.read_retry tries a few times if the first read fails
    humidity, temperature = dht_sensor.read_retry(DHT_SENSOR_TYPE, DHT_GPIO_PIN)

    if humidity is not None and temperature is not None:
        # Round to 1 decimal place
        temp_rounded = round(temperature, 1)
        hum_rounded = round(humidity, 1)

        payload = {
            "temperature": temp_rounded,
            "humidity": hum_rounded
        }

        try:
            requests.post(SERVER_URL, json=payload, timeout=2)
            print(f"Sent → Temp: {temp_rounded}°C | Humidity: {hum_rounded}%")

        except Exception as e:
            print(f"Could not send data: {e}")

    else:
        # This happens sometimes — DHT22 can miss a read occasionally
        print("DHT22 read failed, will retry in 5 seconds...")

    time.sleep(5)  # DHT22 needs at least 2 seconds between reads; 5 is safer
