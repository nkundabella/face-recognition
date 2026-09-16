import cv2
import argparse
import time

from .servo import ServoPanClient


def main():
    parser = argparse.ArgumentParser(description="Validate webcam capture and optional ESP8266 servo pan.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--servo-mqtt", action="store_true", help="Enable MQTT servo control.")
    parser.add_argument("--mqtt-broker", default="broker.benax.rw", help="MQTT broker hostname.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default="face-recognition/servo/pan", help="MQTT servo angle topic.")
    parser.add_argument("--servo-step", type=int, default=5, help="Manual servo angle step.")
    parser.add_argument("--auto-scan", action="store_true", help="Sweep the servo automatically.")
    parser.add_argument("--scan-step", type=int, default=4, help="Angle step for automatic sweeping.")
    parser.add_argument("--scan-every", type=float, default=0.25, help="Seconds between automatic sweep steps.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try changing index (0/1/2).")

    servo = ServoPanClient(
        enabled=args.servo_mqtt,
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
    )
    if servo.enabled:
        servo.center()

    print("Camera test. q=quit")
    if servo.enabled:
        print("Servo controls: a/left=pan left, d/right=pan right, c=center")

    t0 = time.time()
    frames = 0
    fps = 0.0
    scan_dir = 1
    last_scan = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if servo.enabled:
            cv2.putText(
                frame,
                f"Servo angle: {servo.angle} scan={'ON' if args.auto_scan else 'OFF'}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            if args.auto_scan:
                now = time.time()
                if (now - last_scan) >= args.scan_every:
                    next_angle = servo.angle + scan_dir * args.scan_step
                    if next_angle >= servo.cfg.max_angle:
                        next_angle = servo.cfg.max_angle
                        scan_dir = -1
                    elif next_angle <= servo.cfg.min_angle:
                        next_angle = servo.cfg.min_angle
                        scan_dir = 1
                    servo.send_angle(next_angle, force=True)
                    last_scan = now

        cv2.imshow("Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if servo.enabled and key in (ord("a"), 81):
            servo.step(-args.servo_step)
        elif servo.enabled and key in (ord("d"), 83):
            servo.step(args.servo_step)
        elif servo.enabled and key == ord("c"):
            servo.center()

    cap.release()
    servo.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()