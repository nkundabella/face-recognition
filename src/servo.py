# src/servo.py
"""
Optional ESP8266 servo-pan control over MQTT.

Python publishes angle commands to:
  broker.benax.rw:1883
  face-recognition/servo/pan
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except Exception as e:
    mqtt = None
    _MQTT_IMPORT_ERROR = e


@dataclass
class ServoPanConfig:
    min_angle: int = 20
    max_angle: int = 160
    center_angle: int = 90
    kp: float = 8.0
    kd: float = 1.8
    gain: float | None = None  # backward compatibility alias for kp
    deadzone_frac: float = 0.08
    max_step_deg: int = 6
    update_every_s: float = 0.10
    d_filter_alpha: float = 0.6
    reconnect_backoff_s: float = 5.0

    def __post_init__(self):
        if self.gain is not None:
            self.kp = float(self.gain)


class ServoPanClient:
    def __init__(
        self,
        enabled: bool = False,
        cfg: ServoPanConfig | None = None,
        broker: str = "broker.benax.rw",
        port: int = 1883,
        topic: str = "face-recognition/servo/pan",
        client_id: str = "face-recognition-python",
    ):
        self.cfg = cfg or ServoPanConfig()
        self.angle = int(self.cfg.center_angle)
        self.broker = broker
        self.port = int(port)
        self.topic = topic
        self.client_id = client_id
        self._enabled = bool(enabled)
        self._last_send = 0.0
        self._last_error: Optional[str] = None
        self._client: Optional[mqtt.Client] = None

        # PD control tracking state
        self._prev_error: float = 0.0
        self._d_error_filtered: float = 0.0
        self._last_track_time: float = 0.0
        self._last_connect_attempt: float = 0.0

        if self._enabled:
            self._connect()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    def center(self) -> None:
        self._prev_error = 0.0
        self._d_error_filtered = 0.0
        self.send_angle(self.cfg.center_angle, force=True)

    def step(self, delta: int) -> None:
        self.send_angle(self.angle + int(delta), force=True)

    def track_x(self, target_x: float, frame_width: int) -> None:
        if not self.enabled or frame_width <= 0:
            return

        now = time.time()
        frame_center = frame_width * 0.5
        error_frac = (float(target_x) - frame_center) / frame_center

        if abs(error_frac) < self.cfg.deadzone_frac:
            self._prev_error = error_frac
            self._d_error_filtered = 0.0
            self._last_track_time = now
            return

        if self._last_track_time <= 0 or (now - self._last_track_time) > 0.5:
            self._prev_error = error_frac
            self._d_error_filtered = 0.0

        dt = now - self._last_track_time if self._last_track_time > 0 else self.cfg.update_every_s
        if dt < 0.01 or dt > 0.5:
            dt = max(0.01, self.cfg.update_every_s)

        # Derivative calculation: change in error over time
        raw_derivative = (error_frac - self._prev_error) / dt
        # Low-pass filter to reject landmark noise
        alpha = self.cfg.d_filter_alpha
        self._d_error_filtered = alpha * self._d_error_filtered + (1.0 - alpha) * raw_derivative
        self._prev_error = error_frac
        self._last_track_time = now

        # PD control output: P commands direction, D dampens approach to center
        control_output = (self.cfg.kp * error_frac) + (self.cfg.kd * self._d_error_filtered)

        # Slew-rate limiter: prevent violent angle snapping
        max_step = float(self.cfg.max_step_deg)
        clamped_step = max(-max_step, min(max_step, control_output))

        next_angle = self.angle + int(round(clamped_step))
        self.send_angle(next_angle)

    def send_angle(self, angle: int, force: bool = False) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        if not force and (now - self._last_send) < self.cfg.update_every_s:
            return False

        angle = int(max(self.cfg.min_angle, min(self.cfg.max_angle, angle)))
        if self._client is None:
            # Rate-limited non-blocking reconnection attempt
            if (now - self._last_connect_attempt) >= self.cfg.reconnect_backoff_s:
                self._connect()
            else:
                self._last_send = now
                return False

        if self._client is None:
            self._last_send = now
            return False

        info = self._client.publish(self.topic, str(angle), qos=0, retain=False)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.angle = angle
            self._last_send = now
            self._last_error = None
            return True

        self._print_error(f"publish failed with MQTT rc={info.rc}")
        self._last_send = now
        return False

    def _connect(self) -> None:
        self._last_connect_attempt = time.time()
        if mqtt is None:
            raise RuntimeError(
                f"paho-mqtt is required for servo control: {_MQTT_IMPORT_ERROR}\n"
                "Install it with: python -m pip install paho-mqtt"
            )

        try:
            if hasattr(mqtt, "CallbackAPIVersion"):
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
            else:
                client = mqtt.Client(client_id=self.client_id)
            client.connect(self.broker, self.port, keepalive=30)
            client.loop_start()
            self._client = client
            print(f"[servo] MQTT connected: {self.broker}:{self.port}, topic={self.topic}")
        except Exception as e:
            self._print_error(f"MQTT connection failed: {e}")
            self._client = None

    def _print_error(self, msg: str) -> None:
        if msg != self._last_error:
            print(f"[servo] {msg}")
            self._last_error = msg
