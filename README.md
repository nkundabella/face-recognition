# face-recognition-5pt

CPU-only face recognition using Haar face detection, MediaPipe 5-point
landmarks, similarity-transform alignment, and ArcFace ONNX embeddings.

## Setup

Use Python 3.11. MediaPipe's legacy FaceMesh API is not available on Python
3.13.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download a compatible ArcFace ONNX model separately and place it at
`models/embedder_arcface.onnx`. Model weights and face data are intentionally
ignored by Git because they are large and may contain personal information.

Run the pipeline stages from the repository root:

```powershell
python -m src.camera
python -m src.detect
python -m src.landmarks
python -m src.align
python -m src.embed
python -m src.enroll
python -m src.evaluate
python -m src.recognize
```

## Optional ESP8266 servo pan

1. Open `hardware/esp8266_servo_pan/esp8266_servo_pan.ino` in Arduino IDE.
2. Set `WIFI_SSID`, `WIFI_PASS`, and `MQTT_BROKER` for your local network.
3. Install the Arduino `PubSubClient` library and upload to an ESP8266/NodeMCU.
4. Wire the servo signal to D1/GPIO5, power the servo from an external 5 V
   supply, and connect the grounds.
5. Test the Python MQTT control:

```powershell
python -m src.camera --servo-mqtt --auto-scan --mqtt-broker YOUR_BROKER
python -m src.recognize --servo-mqtt
```

The default topic is `face-recognition/servo/pan` and the default port is
`1883`. Do not commit Wi-Fi credentials, broker credentials, enrolled images,
or generated face databases.

