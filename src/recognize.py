# src/recognize.py
"""
Multi-face recognition (CPU-friendly):
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Run: python -m src.recognize
Keys:
  q : quit
  r : reload DB from disk (data/db/face_db.npz)
  +/- : adjust threshold (distance) live
  d : toggle debug overlay
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .embed import DEFAULT_EMBEDDER_MODEL, resolve_embedder_model_path
from .servo import ServoPanClient, ServoPanConfig


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = str(DEFAULT_EMBEDDER_MODEL),
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        model_path = resolve_embedder_model_path(model_path)
        self.model_path = model_path
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)


class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        if haar_xml is None:
            haar_xml = "models/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]

        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]

            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue

            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)

            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue

            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)

            out.append(FaceDet(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0, kps=kps.astype(np.float32)))
        return out


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh

        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )


@dataclass
class TrackedFace:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # (5, 2) float32
    match_result: MatchResult
    aligned: Optional[np.ndarray] = None
    last_embed_time: float = 0.0
    embed_count: int = 0
    lost_frames: int = 0


def _box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def _centroid_dist_norm(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], diag: float) -> float:
    c1x, c1y = 0.5 * (b1[0] + b1[2]), 0.5 * (b1[1] + b1[3])
    c2x, c2y = 0.5 * (b2[0] + b2[2]), 0.5 * (b2[1] + b2[3])
    dist = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
    return dist / diag if diag > 0 else 1.0


class FaceTracker:
    def __init__(
        self,
        reembed_interval: float = 2.0,
        warmup_embeds: int = 3,
        max_lost_frames: int = 15,
        iou_threshold: float = 0.3,
    ):
        self.reembed_interval = float(reembed_interval)
        self.warmup_embeds = int(warmup_embeds)
        self.max_lost_frames = int(max_lost_frames)
        self.iou_threshold = float(iou_threshold)
        self.tracks: Dict[int, TrackedFace] = {}
        self._next_id: int = 1

    def update(
        self,
        frame: np.ndarray,
        detections: List[FaceDet],
        embedder: ArcFaceEmbedderONNX,
        matcher: FaceDBMatcher,
    ) -> List[TrackedFace]:
        now = time.time()
        H, W = frame.shape[:2]
        diag = (W ** 2 + H ** 2) ** 0.5

        det_matched: Dict[int, int] = {}
        track_matched: set[int] = set()

        if self.tracks and detections:
            cost_matrix = []
            track_ids = list(self.tracks.keys())
            for d in detections:
                d_box = (d.x1, d.y1, d.x2, d.y2)
                row = []
                for t_id in track_ids:
                    t = self.tracks[t_id]
                    t_box = (t.x1, t.y1, t.x2, t.y2)
                    iou = _box_iou(d_box, t_box)
                    cdist = _centroid_dist_norm(d_box, t_box, diag)
                    if iou >= self.iou_threshold:
                        score = 1.0 - iou
                    elif cdist < 0.15:
                        score = 1.0 + cdist
                    else:
                        score = 999.0
                    row.append(score)
                cost_matrix.append(row)

            for _ in range(min(len(detections), len(track_ids))):
                min_val = 999.0
                best_d = -1
                best_t = -1
                for d_idx in range(len(detections)):
                    if d_idx in det_matched:
                        continue
                    for t_idx in range(len(track_ids)):
                        t_id = track_ids[t_idx]
                        if t_id in track_matched:
                            continue
                        if cost_matrix[d_idx][t_idx] < min_val:
                            min_val = cost_matrix[d_idx][t_idx]
                            best_d = d_idx
                            best_t = t_id
                if min_val < 500.0 and best_d >= 0 and best_t >= 0:
                    det_matched[best_d] = best_t
                    track_matched.add(best_t)

        active_tracks: List[TrackedFace] = []
        for d_idx, d in enumerate(detections):
            if d_idx in det_matched:
                t_id = det_matched[d_idx]
                track = self.tracks[t_id]
                track.x1 = d.x1
                track.y1 = d.y1
                track.x2 = d.x2
                track.y2 = d.y2
                track.kps = d.kps
                track.lost_frames = 0

                needs_embed = (
                    track.embed_count < self.warmup_embeds
                    or (now - track.last_embed_time) >= self.reembed_interval
                )
                if needs_embed:
                    aligned, _ = align_face_5pt(frame, d.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    track.match_result = matcher.match(emb)
                    track.aligned = aligned
                    track.last_embed_time = now
                    track.embed_count += 1

                active_tracks.append(track)
            else:
                t_id = self._next_id
                self._next_id += 1
                aligned, _ = align_face_5pt(frame, d.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)

                new_track = TrackedFace(
                    track_id=t_id,
                    x1=d.x1,
                    y1=d.y1,
                    x2=d.x2,
                    y2=d.y2,
                    kps=d.kps,
                    match_result=mr,
                    aligned=aligned,
                    last_embed_time=now,
                    embed_count=1,
                    lost_frames=0,
                )
                self.tracks[t_id] = new_track
                active_tracks.append(new_track)

        unmatched_tracks = set(self.tracks.keys()) - track_matched - {t.track_id for t in active_tracks}
        for t_id in list(unmatched_tracks):
            self.tracks[t_id].lost_frames += 1
            if self.tracks[t_id].lost_frames > self.max_lost_frames:
                del self.tracks[t_id]

        return active_tracks


def main():
    parser = argparse.ArgumentParser(description="Live face recognition with optional ESP8266 servo pan tracking.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--servo-mqtt", action="store_true", help="Enable MQTT servo control.")
    parser.add_argument("--mqtt-broker", default="broker.benax.rw", help="MQTT broker hostname.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default="face-recognition/servo/pan", help="MQTT servo angle topic.")
    parser.add_argument("--servo-min", type=int, default=20, help="Minimum safe servo angle.")
    parser.add_argument("--servo-max", type=int, default=160, help="Maximum safe servo angle.")
    parser.add_argument("--servo-center", type=int, default=90, help="Centered servo angle.")
    parser.add_argument("--servo-gain", type=float, default=10.0, help="Pan correction strength.")
    parser.add_argument("--servo-deadzone", type=float, default=0.08, help="Ignore small horizontal face offsets.")
    parser.add_argument("--servo-step", type=int, default=5, help="Manual servo angle step.")
    parser.add_argument("--servo-scan-step", type=int, default=4, help="Angle step while searching for a face.")
    parser.add_argument("--servo-scan-every", type=float, default=0.25, help="Seconds between search-scan steps.")
    parser.add_argument("--servo-scan-after", type=float, default=0.4, help="Seconds without a face before scanning.")
    parser.add_argument(
        "--servo-lost-after",
        type=float,
        default=1.5,
        help="Seconds to hold the last angle after losing the locked identity before searching.",
    )
    parser.add_argument(
        "--identify-wait",
        type=float,
        default=2.0,
        help="Seconds to pause on a face while confirming its identity before tracking or scanning again.",
    )
    parser.add_argument(
        "--reembed-interval",
        type=float,
        default=2.0,
        help="Seconds between ArcFace re-embedding refreshes for tracked faces.",
    )
    parser.add_argument(
        "--warmup-embeds",
        type=int,
        default=3,
        help="Number of initial frames to embed a new face before caching its identity.",
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=15,
        help="Number of consecutive missing frames before dropping a tracked face.",
    )
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_EMBEDDER_MODEL, input_size=(112, 112), debug=False)

    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.24)
    tracker = FaceTracker(
        reembed_interval=args.reembed_interval,
        warmup_embeds=args.warmup_embeds,
        max_lost_frames=args.max_lost_frames,
    )

    servo = ServoPanClient(
        enabled=args.servo_mqtt,
        cfg=ServoPanConfig(
            min_angle=args.servo_min,
            max_angle=args.servo_max,
            center_angle=args.servo_center,
            gain=args.servo_gain,
            deadzone_frac=args.servo_deadzone,
        ),
        broker=args.mqtt_broker,
        port=args.mqtt_port,
        topic=args.mqtt_topic,
    )
    servo_tracking = servo.enabled
    if servo.enabled:
        servo.center()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Recognize (multi-face). q=quit, r=reload DB, +/- threshold, d=debug overlay")
    if servo.enabled:
        print("Servo: t=toggle tracking, left/right=manual pan, c=center")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    servo_scan_dir = 1
    last_scan = 0.0
    last_seen_face = time.time()
    last_seen_target = time.time()
    face_pause_started: Optional[float] = None
    candidate_name: Optional[str] = None
    candidate_started = 0.0
    confirmed_name: Optional[str] = None
    locked_name: Optional[str] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        h, w = vis.shape[:2]
        thumb = 112
        pad = 8
        x0 = w - thumb - pad
        y0 = 80
        shown = 0
        track_x: Optional[float] = None
        largest_index: Optional[int] = None
        match_results: List[MatchResult] = []

        tracked_faces = tracker.update(frame, faces, embedder, matcher)

        for i, tf in enumerate(tracked_faces):
            mr = tf.match_result
            match_results.append(mr)

            cv2.rectangle(vis, (tf.x1, tf.y1), (tf.x2, tf.y2), (0, 255, 0), 2)
            for (x, y) in tf.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            label = mr.name if mr.name is not None else "Unknown"
            line1 = f"#{tf.track_id} {label}"
            line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"

            color = (0, 255, 0) if mr.accepted else (0, 0, 255)
            cv2.putText(vis, line1, (tf.x1, max(0, tf.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(vis, line2, (tf.x1, max(0, tf.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if tf.aligned is not None and y0 + thumb <= h and shown < 4:
                vis[y0:y0 + thumb, x0:x0 + thumb] = tf.aligned
                cv2.putText(vis, f"#{tf.track_id}:{label}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y0 += thumb + pad
                shown += 1

            if show_debug:
                dbg = f"kpsLeye=({tf.kps[0,0]:.0f},{tf.kps[0,1]:.0f})"
                cv2.putText(vis, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        now = time.time()
        target_visible = False
        if tracked_faces:
            largest_index = max(
                range(len(tracked_faces)),
                key=lambda i: (tracked_faces[i].x2 - tracked_faces[i].x1) * (tracked_faces[i].y2 - tracked_faces[i].y1),
            )
            largest = tracked_faces[largest_index]
            largest_match = match_results[largest_index]
            if locked_name is not None:
                target_index = next(
                    (i for i, result in enumerate(match_results) if result.accepted and result.name == locked_name),
                    None,
                )
                if target_index is not None:
                    target = tracked_faces[target_index]
                    target_visible = True
                    last_seen_face = now
                    last_seen_target = now
                    track_x = 0.5 * float(target.x1 + target.x2)
                else:
                    candidate_name = None
                    confirmed_name = None
            else:
                if face_pause_started is None:
                    face_pause_started = now
                    candidate_name = None
                    confirmed_name = None

                if largest_match.accepted and largest_match.name is not None:
                    if largest_match.name != candidate_name:
                        candidate_name = largest_match.name
                        candidate_started = now
                        confirmed_name = None
                    elif (now - candidate_started) >= args.identify_wait:
                        confirmed_name = candidate_name
                        locked_name = confirmed_name
                        target_visible = True
                        last_seen_face = now
                        last_seen_target = now
                        track_x = 0.5 * float(largest.x1 + largest.x2)
                elif (now - face_pause_started) >= args.identify_wait:
                    candidate_name = None
                    confirmed_name = None
        else:
            if locked_name is None:
                face_pause_started = None
                candidate_name = None
                confirmed_name = None

        if servo_tracking and servo.enabled:
            if target_visible and track_x is not None:
                servo.track_x(track_x, w)
            else:
                confirmation_started = candidate_started if candidate_name is not None else face_pause_started
                identify_done = confirmation_started is None or (now - confirmation_started) >= args.identify_wait
                no_target_timeout = (now - last_seen_target) >= args.servo_lost_after
                if identify_done and no_target_timeout:
                    if (now - last_scan) >= args.servo_scan_every:
                        next_angle = servo.angle + servo_scan_dir * args.servo_scan_step
                        if next_angle >= servo.cfg.max_angle:
                            next_angle = servo.cfg.max_angle
                            servo_scan_dir = -1
                        elif next_angle <= servo.cfg.min_angle:
                            next_angle = servo.cfg.min_angle
                            servo_scan_dir = 1
                        servo.send_angle(next_angle, force=True)
                        last_scan = now

        header = f"IDs={len(matcher._names)} thr(dist)={matcher.dist_thresh:.2f}"
        if fps is not None:
            header += f" fps={fps:.1f}"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if servo.enabled:
            if target_visible:
                mode = "TRACK"
            elif locked_name is not None:
                mode = (
                    f"SEARCH {locked_name}"
                    if (now - last_seen_target) >= args.servo_lost_after
                    else f"HOLD {locked_name}"
                )
            elif tracked_faces and face_pause_started is not None and (now - face_pause_started) < args.identify_wait:
                mode = f"IDENTIFY {now - face_pause_started:.1f}s"
            else:
                mode = "SCAN"
            if not servo_tracking:
                mode = "MANUAL"
            servo_line = f"servo={servo.angle} mode={mode}"
            cv2.putText(vis, servo_line, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
            if track_x is not None:
                cv2.line(vis, (int(track_x), 0), (int(track_x), h), (255, 255, 0), 1)

        cv2.imshow("recognize_new", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif servo.enabled and key == ord("t"):
            servo_tracking = not servo_tracking
            print(f"[servo] tracking: {'ON' if servo_tracking else 'OFF'}")
        elif servo.enabled and key in (81,):
            servo_tracking = False
            servo.step(-args.servo_step)
        elif servo.enabled and key in (83,):
            servo_tracking = False
            servo.step(args.servo_step)
        elif servo.enabled and key == ord("c"):
            servo.center()
        elif key == ord("r"):
            matcher.reload_from(db_path)
            print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("-"):
            matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")

    cap.release()
    servo.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
