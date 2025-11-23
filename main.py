import os, math, csv
from collections import defaultdict, deque
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "VIDEO_FOLDER": "data/videos",
    "ANNOT_FOLDER": "annotations",
    "OUTPUT_FOLDER": "output",
    "MIN_AREA": 8,
    "MAX_AREA": 5000,
    "FEATURE_HISTORY": 12,
    "TRACK_MAX_AGE": 8,
    "STD_MULT": 2.0,
    "STRAIGHTNESS_LOW_PERC": 5,
}

os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)

# ============================================================
# UTILITY
# ============================================================
def centroid_from_box(box):
    x, y, w, h = box
    return (int(x + w/2), int(y + h/2))

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def features_from_history(history):
    """
    history: list of (cx, cy, w, h)
    returns: 7D feature vector
    """
    if len(history) < 4:
        return None

    pts = np.array([[h[0], h[1]] for h in history], float)
    sizes = np.array([h[2] * h[3] for h in history], float)

    vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean_speed = float(np.mean(vel))
    var_speed  = float(np.var(vel))
    smoothness = float(np.mean(np.abs(np.diff(vel)))) if len(vel) > 1 else 0.0

    path_len = float(np.sum(vel)) + 1e-9
    end2end  = float(np.linalg.norm(pts[-1] - pts[0]))
    straightness = end2end / path_len

    mean_size = float(np.mean(sizes))
    var_size  = float(np.var(sizes))
    bbox_jitter = float(np.mean(np.abs(np.diff(sizes))) / (mean_size + 1e-9))

    return np.array([mean_speed, var_speed, smoothness,
                     mean_size, var_size, straightness, bbox_jitter])

# ============================================================
# ANNOTATION LOADING
# ============================================================
def load_annotations_from_folder(folder):
    """
    Reads: frame track x y w h label
    Only keeps label 'drone'
    """
    if not os.path.exists(folder):
        print("[Ann] Folder not found:", folder)
        return []

    txts = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    if not txts:
        print("[Ann] No annotation files found.")
        return []

    drone_tracks = []

    for tfile in txts:
        path = os.path.join(folder, tfile)
        per_tid = {}

        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 7:
                    continue

                frame = int(parts[0])
                tid   = int(parts[1])
                x     = int(parts[2])
                y     = int(parts[3])
                w     = int(parts[4])
                h     = int(parts[5])
                label = parts[6].lower()

                if label != "drone":
                    continue

                cx = x + w/2
                cy = y + h/2

                if tid not in per_tid:
                    per_tid[tid] = []
                per_tid[tid].append((cx, cy, w, h))

        for tid, hist in per_tid.items():
            if len(hist) >= 4:
                drone_tracks.append({"hist": hist, "label": "drone"})

    print(f"[Ann] Loaded {len(drone_tracks)} drone tracks.")
    return drone_tracks

# ============================================================
# LEARN DRONE THRESHOLDS
# ============================================================
def learn_thresholds(tracks):
    feats = []
    for t in tracks:
        f = features_from_history(t["hist"])
        if f is not None:
            feats.append(f)

    if not feats:
        raise RuntimeError("No annotated tracks contain enough frames.")

    F = np.vstack(feats)
    names = ["mean_speed","var_speed","smoothness",
             "mean_size","var_size","straightness","bbox_jitter"]

    stats = {}
    for i, name in enumerate(names):
        col = F[:, i]
        mean = float(np.mean(col))
        std  = float(np.std(col))

        if name == "straightness":
            low  = max(0, np.percentile(col, CONFIG["STRAIGHTNESS_LOW_PERC"]))
            high = 1.0
        elif name == "bbox_jitter":
            low  = 0
            high = mean + CONFIG["STD_MULT"] * std
        else:
            low  = mean - CONFIG["STD_MULT"] * std
            high = mean + CONFIG["STD_MULT"] * std
            if low < 0:
                low = 0

        stats[name] = {"low": low, "high": high}

    print("\n[Learn] Learned feature ranges:")
    for n, v in stats.items():
        print(f"   {n}: low={v['low']:.4f}, high={v['high']:.4f}")
    return stats

# ============================================================
# DECISION FUNCTION
# ============================================================
def decide_is_drone(feature, thresholds):
    if feature is None:
        return False

    names = ["mean_speed","var_speed","smoothness",
             "mean_size","var_size","straightness","bbox_jitter"]

    score = 0
    for i, name in enumerate(names):
        val = feature[i]
        low = thresholds[name]["low"]
        high = thresholds[name]["high"]

        if name == "straightness":
            if val >= low: score += 1
        elif name == "bbox_jitter":
            if val <= high: score += 1
        else:
            if low <= val <= high: score += 1

    return score >= 4  # majority

# ============================================================
# TRACKER
# ============================================================
class TrackObj:
    def __init__(self, box, tid):
        cx, cy = centroid_from_box(box)
        self.id = tid
        self.bbox = box
        self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])
        self.history.append((cx, cy, box[2], box[3]))
        self.time_since_update = 0
        self.kf = self._init_kf(cx, cy)

    def _init_kf(self, cx, cy):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        kf.x = np.array([cx, cy, 0, 0], float)
        kf.P *= 500
        kf.R *= 5
        kf.Q = np.eye(4) * 0.01
        return kf

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box):
        cx, cy = centroid_from_box(box)
        self.kf.update([cx,cy])
        self.bbox = box
        self.history.append((cx,cy,box[2],box[3]))
        self.time_since_update = 0

    def pos(self):
        return int(self.kf.x[0]), int(self.kf.x[1])

class TrackerManager:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, boxes):
        for t in self.tracks:
            t.predict()
        if not boxes:
            self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
            return self.tracks

        det_centers = [centroid_from_box(b) for b in boxes]
        track_centers = [t.pos() for t in self.tracks]

        if track_centers:
            cost = np.zeros((len(track_centers), len(det_centers)))
            for i, tc in enumerate(track_centers):
                for j, dc in enumerate(det_centers):
                    cost[i,j] = math.hypot(tc[0]-dc[0], tc[1]-dc[1])

            row, col = linear_sum_assignment(cost)

            matched_det = set()
            matched_tr  = set()

            for r, c in zip(row, col):
                if cost[r,c] < 60:
                    self.tracks[r].update(boxes[c])
                    matched_tr.add(r)
                    matched_det.add(c)

            for j, b in enumerate(boxes):
                if j not in matched_det:
                    self.tracks.append(TrackObj(b, self.next_id))
                    self.next_id += 1
        else:
            for b in boxes:
                self.tracks.append(TrackObj(b, self.next_id))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
        return self.tracks

# ============================================================
# MOTION DETECTOR
# ============================================================
class MotionDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=16, detectShadows=False)

    def get_candidates(self, frame):
        mask = self.bg.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if CONFIG["MIN_AREA"] <= area <= CONFIG["MAX_AREA"]:
                x,y,w,h = cv2.boundingRect(c)
                boxes.append((x,y,w,h))
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        return boxes, mask

# ============================================================
# PROCESS VIDEO
# ============================================================
def process_video(video_path, thresholds, visualize=False):
    print("\n[Run] Processing:", video_path)
    cap = cv2.VideoCapture(video_path)

    # ---- READ FIRST FRAME SAFELY ----
    ret, first_frame = cap.read()
    if not ret:
        print("[Run] ERROR: cannot read video.")
        return

    # Ensure BGR
    if len(first_frame.shape) == 2:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)

    height, width = first_frame.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 20.0

    out_path = os.path.join(CONFIG["OUTPUT_FOLDER"],
                            os.path.basename(video_path).rsplit(".",1)[0] + "_annotated.mp4")

    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (width, height))

    md = MotionDetector()
    tracker = TrackerManager()

    # ---- PROCESS FIRST FRAME ----
    def process_frame(frame, frame_idx):
        boxes, _ = md.get_candidates(frame)
        tracks = tracker.update(boxes)

        for t in tracks:
            feat = features_from_history(list(t.history))
            is_dr = decide_is_drone(feat, thresholds)

            x,y,w,h = t.bbox
            color = (0,0,255) if is_dr else (0,255,0)
            label = "DRONE" if is_dr else "OTHER"
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if is_dr:
                cx, cy = centroid_from_box(t.bbox)
                print(f"[ALARM] F{frame_idx}: DRONE at ({cx},{cy})")

        writer.write(frame)

    # Process first frame
    process_frame(first_frame, 0)

    # ---- PROCESS REMAINING FRAMES ----
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        process_frame(frame, frame_idx)
        frame_idx += 1

        if visualize:
            cv2.imshow("out", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    if visualize:
        cv2.destroyAllWindows()

    print("[Run] Saved:", out_path)

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    tracks = load_annotations_from_folder(CONFIG["ANNOT_FOLDER"])
    if not tracks:
        print("[Main] No valid drone tracks. Exiting.")
        return

    thresholds = learn_thresholds(tracks)

    vids = [
        os.path.join(CONFIG["VIDEO_FOLDER"], v)
        for v in os.listdir(CONFIG["VIDEO_FOLDER"])
        if v.lower().endswith((".mp4",".avi",".mov",".mkv"))
    ]

    if not vids:
        print("[Main] No videos found.")
        return

    for v in vids:
        process_video(v, thresholds, visualize=False)

# ============================================================
if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# drone_rule_autolearn.py

# Option A: Rule-based drone detector with auto-learned thresholds from drone annotations.

# Assumptions:
# - Annotations are in folder: ./annotations/*.txt
# - Annotation line format (space-separated):
#     frame_idx track_id x y w h label
#   e.g.:
#     0 1 708 757 15 13 drone
# - Videos are in: ./data/videos/*.mp4 (or avi/mov/mkv)
# - Output annotated videos saved to ./output/

# Run:
#     python drone_rule_autolearn.py
# """

# import os, math, csv
# from collections import defaultdict, deque
# import numpy as np
# import cv2
# from filterpy.kalman import KalmanFilter
# from scipy.optimize import linear_sum_assignment

# # ---------- CONFIG ----------
# CONFIG = {
#     "VIDEO_FOLDER": "data/videos",
#     "ANNOT_FOLDER": "annotations",
#     "OUTPUT_FOLDER": "output",
#     "MIN_AREA": 8,
#     "MAX_AREA": 5000,
#     "FEATURE_HISTORY": 12,
#     "TRACK_MAX_AGE": 8,
#     # thresholds learning parameters:
#     "STD_MULT": 2.0,   # threshold = mean ± STD_MULT * std  (for appropriate features)
#     "STRAIGHTNESS_LOW_PERC": 5,  # use percentiles for some one-sided thresholds
# }

# os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)

# # ---------- Utilities ----------
# def centroid_from_box(box):
#     x, y, w, h = box
#     return (int(x + w/2), int(y + h/2))

# # ---------- Feature extraction ----------
# def features_from_history(history):
#     """
#     history: list of (cx, cy, w, h), ordered by time
#     Returns feature vector:
#       [mean_speed, var_speed, smoothness, mean_size, var_size, straightness, bbox_jitter]
#     """
#     if len(history) < 4:
#         return None
#     pts = np.array([[h[0], h[1]] for h in history], dtype=float)
#     sizes = np.array([h[2]*h[3] for h in history], dtype=float)

#     # velocities
#     vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # per-frame displacement
#     mean_speed = float(np.mean(vel))
#     var_speed = float(np.var(vel))
#     smoothness = float(np.mean(np.abs(np.diff(vel)))) if len(vel) > 1 else 0.0

#     # straightness: end-to-end / path length (1.0 is perfectly straight)
#     path_len = float(np.sum(vel)) + 1e-9
#     end2end = float(np.linalg.norm(pts[-1] - pts[0]))
#     straightness = end2end / path_len

#     mean_size = float(np.mean(sizes))
#     var_size = float(np.var(sizes))

#     # bounding-box jitter: mean frame-to-frame size change relative to mean size
#     bbox_diff = np.abs(np.diff(sizes))
#     bbox_jitter = float(np.mean(bbox_diff) / (mean_size + 1e-9))

#     return np.array([mean_speed, var_speed, smoothness, mean_size, var_size, straightness, bbox_jitter])

# # ---------- Annotation loader ----------
# def load_annotations_from_folder(ann_folder):
#     """
#     Reads all .txt files in ann_folder.
#     Each file: lines like "frame track_id x y w h label" (space-separated).
#     Returns list of track histories (one per annotated track), each: {'hist': [(cx,cy,w,h),...], 'label': label}
#     """
#     if not os.path.exists(ann_folder):
#         print("[Ann] Annotation folder not found:", ann_folder)
#         return []

#     txts = [f for f in os.listdir(ann_folder) if f.lower().endswith(".txt")]
#     if not txts:
#         print("[Ann] No .txt annotation files in", ann_folder)
#         return []

#     all_tracks = []
#     for tfile in txts:
#         path = os.path.join(ann_folder, tfile)
#         tracks = {}  # track_id -> {'hist': [], 'label': label}
#         with open(path, "r", encoding="utf-8", errors="ignore") as fh:
#             for line in fh:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 parts = line.split()
#                 # Expect exactly 7 parts: frame tid x y w h label
#                 if len(parts) != 7:
#                     continue
#                 try:
#                     frame = int(parts[0])
#                     tid = int(parts[1])
#                     x = int(parts[2]); y = int(parts[3]); w = int(parts[4]); h = int(parts[5])
#                     label = parts[6].strip().lower()
#                 except Exception:
#                     continue

#                 cx = x + w/2.0
#                 cy = y + h/2.0
#                 if tid not in tracks:
#                     tracks[tid] = {'hist': [], 'label': label}
#                 tracks[tid]['hist'].append((cx, cy, w, h))

#         # Keep tracks with enough history and the label is drone
#         for tid, val in tracks.items():
#             if val['label'].startswith("drone") and len(val['hist']) >= 4:
#                 all_tracks.append({'hist': val['hist'], 'label': 'drone', 'source_file': tfile, 'track_id': tid})
#     print(f"[Ann] Loaded {len(all_tracks)} drone tracks from {len(txts)} files.")
#     return all_tracks

# # ---------- Compute thresholds from drone tracks ----------
# def learn_thresholds_from_tracks(tracks):
#     """
#     Input: list of track dicts with 'hist'
#     Computes features per track and returns threshold dict.
#     We compute mean/std for each numeric feature and set allowed ranges.
#     """
#     feats = []
#     for t in tracks:
#         f = features_from_history(t['hist'])
#         if f is not None:
#             feats.append(f)
#     if len(feats) == 0:
#         raise RuntimeError("No valid annotated tracks to learn from.")

#     F = np.vstack(feats)  # shape (N_tracks, n_features)
#     names = ['mean_speed','var_speed','smoothness','mean_size','var_size','straightness','bbox_jitter']
#     stats = {}
#     for i, n in enumerate(names):
#         col = F[:, i]
#         mean = float(np.mean(col))
#         std = float(np.std(col))
#         # compute symmetric bounds for many features; for straightness and bbox_jitter, one-sided makes sense
#         if n in ('straightness',):
#             low = max(0.0, np.percentile(col, CONFIG["STRAIGHTNESS_LOW_PERC"]))
#             high = 1.0  # straightness upper bound capped
#         elif n in ('bbox_jitter',):
#             # jitter should be small; use high bound
#             low = 0.0
#             high = mean + CONFIG["STD_MULT"] * std
#         else:
#             low = mean - CONFIG["STD_MULT"] * std
#             high = mean + CONFIG["STD_MULT"] * std
#             # prevent negative lower bounds where impossible
#             if low < 0 and n not in ('mean_size', 'var_size'):
#                 low = 0.0
#         stats[n] = {'mean': mean, 'std': std, 'low': low, 'high': high}

#     print("[Learn] Learned thresholds (per feature):")
#     for n, v in stats.items():
#         print(f"    {n}: mean={v['mean']:.4g}, std={v['std']:.4g}, low={v['low']:.4g}, high={v['high']:.4g}")
#     return stats

# # ---------- Detector decision using learned thresholds ----------
# def decide_is_drone(feat, thresholds):
#     """
#     feat: array as returned by features_from_history
#     thresholds: dict from learn_thresholds_from_tracks
#     Returns True if feat falls inside drone-like ranges (i.e., matches drone signature)
#     """
#     if feat is None:
#         return False
#     names = ['mean_speed','var_speed','smoothness','mean_size','var_size','straightness','bbox_jitter']
#     score = 0
#     total = len(names)
#     for i, n in enumerate(names):
#         val = float(feat[i])
#         low = thresholds[n]['low']
#         high = thresholds[n]['high']
#         # check membership; for features with one-sided bounds we treat appropriately
#         if n == 'straightness':
#             # require > low
#             if val >= low:
#                 score += 1
#         elif n == 'bbox_jitter':
#             # require val <= high
#             if val <= high:
#                 score += 1
#         else:
#             if val >= low and val <= high:
#                 score += 1
#     # require a majority of features to match (tunable)
#     return (score >= math.ceil(0.6 * total))

# # ---------- Minimal Kalman-based tracker & tracker manager ----------
# class TrackObj:
#     def __init__(self, box, tid):
#         cx, cy = centroid_from_box(box)
#         self.bbox = box
#         self.id = tid
#         self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])
#         self.history.append((cx, cy, box[2], box[3]))
#         self.kf = self._init_kf(cx, cy)
#         self.time_since_update = 0

#     def _init_kf(self, cx, cy):
#         kf = KalmanFilter(dim_x=4, dim_z=2)
#         kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], float)
#         kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
#         kf.x = np.array([cx, cy, 0, 0], float)
#         kf.P *= 500.
#         kf.R *= 5.
#         kf.Q = np.eye(4) * 0.01
#         return kf

#     def predict(self):
#         self.kf.predict()
#         self.time_since_update += 1

#     def update(self, box):
#         cx, cy = centroid_from_box(box)
#         self.kf.update(np.array([cx, cy]))
#         self.bbox = box
#         self.history.append((cx, cy, box[2], box[3]))
#         self.time_since_update = 0

#     def pos(self):
#         return int(self.kf.x[0]), int(self.kf.x[1])

# class TrackerManager:
#     def __init__(self):
#         self.tracks = []
#         self.next_id = 0

#     def update(self, boxes):
#         # predict
#         for t in self.tracks:
#             t.predict()
#         if len(boxes) == 0:
#             # prune old
#             self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
#             return self.tracks

#         det_centers = [centroid_from_box(b) for b in boxes]
#         track_centers = [t.pos() for t in self.tracks] if self.tracks else []

#         if track_centers:
#             cost = np.zeros((len(track_centers), len(det_centers)))
#             for i, tc in enumerate(track_centers):
#                 for j, dc in enumerate(det_centers):
#                     cost[i, j] = math.hypot(tc[0]-dc[0], tc[1]-dc[1])
#             row, col = linear_sum_assignment(cost)
#             matched_det = set()
#             matched_track = set()
#             for r, c in zip(row, col):
#                 if cost[r, c] < 60:
#                     self.tracks[r].update(boxes[c])
#                     matched_track.add(r)
#                     matched_det.add(c)
#             # new detections -> new tracks
#             for j, b in enumerate(boxes):
#                 if j not in matched_det:
#                     self.tracks.append(TrackObj(b, self.next_id)); self.next_id += 1
#         else:
#             for b in boxes:
#                 self.tracks.append(TrackObj(b, self.next_id)); self.next_id += 1

#         # remove stale
#         self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
#         return self.tracks

# # ---------- Motion detector ----------
# class MotionDetector:
#     def __init__(self):
#         self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16, detectShadows=False)

#     def get_candidates(self, frame):
#         mask = self.bg.apply(frame)
#         mask = cv2.medianBlur(mask, 5)
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         boxes = []
#         for c in cnts:
#             area = cv2.contourArea(c)
#             if area < CONFIG["MIN_AREA"] or area > CONFIG["MAX_AREA"]:
#                 continue
#             x, y, w, h = cv2.boundingRect(c)
#             boxes.append((x, y, w, h))
#         # sort by area descending
#         boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
#         return boxes, mask

# # ---------- Run detection on a single video ----------
# def process_video_rule(video_path, thresholds, visualize=False):
#     cap = cv2.VideoCapture(video_path)
#     md = MotionDetector()
#     mgr = TrackerManager()
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

#     out_path = os.path.join("output", os.path.basename(video_path).rsplit(".",1)[0] + "_annotated.mp4")
#     writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#     frame_idx = 0
#     print("[Run] Processing:", video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         boxes, mask = md.get_candidates(frame)
#         tracks = mgr.update(boxes)

#         for t in tracks:
#             feat = features_from_history(list(t.history))
#             is_dr = decide_is_drone(feat, thresholds)
#             x, y, w, h = t.bbox
#             color = (0,0,255) if is_dr else (0,255,0)
#             label = "DRONE" if is_dr else "OTHER"
#             cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
#             cv2.putText(frame, f"{label}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
#             if is_dr:
#                 cx, cy = centroid_from_box(t.bbox)
#                 print(f"[ALARM] Frame {frame_idx}: DRONE detected at track {t.id} pos=({cx},{cy})")

#         writer.write(frame)
#         if visualize:
#             cv2.imshow("frame", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#         frame_idx += 1

#     cap.release()
#     writer.release()
#     if visualize:
#         cv2.destroyAllWindows()
#     print("[Run] Saved annotated video to:", out_path)

# # ---------- Main ----------
# def main():
#     # 1) load annotated drone tracks
#     ann_folder = CONFIG["ANNOT_FOLDER"]
#     tracks = load_annotations_from_folder(ann_folder)
#     if not tracks:
#         print("[Main] No annotated drone tracks found. Exiting.")
#         return

#     # 2) learn thresholds
#     thresholds = learn_thresholds_from_tracks(tracks)

#     # 3) process each video in data/videos
#     vfolder = CONFIG["VIDEO_FOLDER"]
#     if not os.path.exists(vfolder):
#         print("[Main] Video folder not found:", vfolder); return
#     vids = [os.path.join(vfolder,f) for f in os.listdir(vfolder) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
#     if not vids:
#         print("[Main] No videos found in", vfolder); return

#     for v in vids:
#         process_video_rule(v, thresholds, visualize=False)

# if __name__ == "__main__":
#     main()
# # # #!/usr/bin/env python3
# # # """
# # # FULLY AUTOMATIC DRONE VS BIRD PIPELINE
# # # --------------------------------------

# # # What this script does AUTOMATICALLY:

# # # 1. Creates required folders:
# # #        data/videos
# # #        data/frames
# # #        data/annotations.csv (template)
# # #        models/

# # # 2. Extracts frames from all videos in data/videos/

# # # 3. If annotations.csv exists and has >=10 samples:
# # #        → trains SVM
# # #        → saves SVM to models/svm_drone_bird.pkl

# # # 4. If model exists:
# # #        → loads model
# # #    else:
# # #        → uses rule-based detector

# # # 5. Runs full drone-vs-bird detection on EACH video
# # #        → Saves output video to output/ folder

# # # NO USER ACTION REQUIRED.
# # # --------------------------------------
# # # """

# # # import os
# # # import cv2
# # # import numpy as np
# # # from collections import deque
# # # from sklearn.svm import SVC
# # # from sklearn.preprocessing import StandardScaler
# # # import joblib
# # # from filterpy.kalman import KalmanFilter
# # # from scipy.optimize import linear_sum_assignment
# # # import csv
# # # import math

# # # # ---------------------------------------------------------
# # # # CONFIGURATION (DO NOT CHANGE ANYTHING)
# # # # ---------------------------------------------------------
# # # CONFIG = {
# # #     "VIDEO_FOLDER": "data/videos",
# # #     "FRAMES_FOLDER": "data/frames",
# # #     "ANNOTATIONS_CSV": "data/annotations.csv",
# # #     "MODEL_OUT": "models/svm_drone_bird.pkl",
# # #     "OUTPUT_FOLDER": "output",
# # #     "FRAME_STEP": 5,
# # #     "MIN_AREA": 10,
# # #     "MAX_AREA": 5000,
# # #     "FEATURE_HISTORY": 12,
# # #     "TRACK_MAX_AGE": 8,
# # # }

# # # # ---------------------------------------------------------
# # # # Ensure all folders exist
# # # # ---------------------------------------------------------
# # # for folder in [CONFIG["VIDEO_FOLDER"], CONFIG["FRAMES_FOLDER"], 
# # #                os.path.dirname(CONFIG["MODEL_OUT"]), CONFIG["OUTPUT_FOLDER"]]:
# # #     if not os.path.exists(folder):
# # #         os.makedirs(folder)

# # # # ---------------------------------------------------------
# # # # Utility: centroid
# # # # ---------------------------------------------------------
# # # def centroid_from_box(b):
# # #     x, y, w, h = b
# # #     return (int(x + w/2), int(y + h/2))

# # # # ---------------------------------------------------------
# # # # FRAME EXTRACTION (AUTOMATIC)
# # # # ---------------------------------------------------------
# # # def extract_frames(video_path, out_folder, step=5):
# # #     os.makedirs(out_folder, exist_ok=True)
# # #     cam = cv2.VideoCapture(video_path)
# # #     idx = 0
# # #     saved = 0
# # #     while True:
# # #         ret, frame = cam.read()
# # #         if not ret:
# # #             break
# # #         if idx % step == 0:
# # #             cv2.imwrite(os.path.join(out_folder, f"frame_{idx}.jpg"), frame)
# # #             saved += 1
# # #         idx += 1
# # #     cam.release()
# # #     print(f"[Frames] Saved {saved} frames to {out_folder}")

# # # # ---------------------------------------------------------
# # # # MOTION DETECTOR
# # # # ---------------------------------------------------------
# # # class MotionDetector:
# # #     def __init__(self):
# # #         self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16)

# # #     def get_candidates(self, frame):
# # #         mask = self.bg.apply(frame)
# # #         mask = cv2.medianBlur(mask, 5)
# # #         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# # #         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # #         boxes = []
# # #         for c in cnts:
# # #             area = cv2.contourArea(c)
# # #             if CONFIG["MIN_AREA"] < area < CONFIG["MAX_AREA"]:
# # #                 x,y,w,h = cv2.boundingRect(c)
# # #                 boxes.append((x,y,w,h))
# # #         return boxes, mask

# # # # ---------------------------------------------------------
# # # # TRACKER
# # # # ---------------------------------------------------------
# # # class Track:
# # #     def __init__(self, box, id):
# # #         cx, cy = centroid_from_box(box)
# # #         self.id = id
# # #         self.bbox = box
# # #         self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])
# # #         self.history.append((cx,cy,box[2],box[3]))
# # #         self.kf = self.init_kf(cx, cy)
# # #         self.time_since_update = 0

# # #     def init_kf(self, cx, cy):
# # #         kf = KalmanFilter(dim_x=4, dim_z=2)
# # #         kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
# # #         kf.H = np.array([[1,0,0,0],[0,1,0,0]])
# # #         kf.x = np.array([cx,cy,0,0])
# # #         kf.P *= 500
# # #         kf.R *= 5
# # #         kf.Q = np.eye(4)*0.01
# # #         return kf

# # #     def predict(self):
# # #         self.kf.predict()
# # #         self.time_since_update += 1

# # #     def update(self, box):
# # #         cx, cy = centroid_from_box(box)
# # #         self.kf.update(np.array([cx,cy]))
# # #         self.bbox = box
# # #         self.history.append((cx,cy,box[2],box[3]))
# # #         self.time_since_update = 0

# # #     def pos(self):
# # #         return int(self.kf.x[0]), int(self.kf.x[1])

# # # class TrackerManager:
# # #     def __init__(self):
# # #         self.tracks = []
# # #         self.next_id = 0

# # #     def update(self, boxes):
# # #         centers = [centroid_from_box(b) for b in boxes]

# # #         # predict
# # #         for t in self.tracks:
# # #             t.predict()

# # #         if len(centers) == 0:
# # #             return self.tracks

# # #         # cost matrix
# # #         track_pos = [t.pos() for t in self.tracks]
# # #         cost = np.zeros((len(track_pos), len(centers)))
# # #         for i,tp in enumerate(track_pos):
# # #             for j,c in enumerate(centers):
# # #                 cost[i,j] = math.hypot(tp[0]-c[0], tp[1]-c[1])

# # #         # assign
# # #         row,col = linear_sum_assignment(cost)

# # #         matched_tracks = set()
# # #         matched_det = set()

# # #         for r,c in zip(row, col):
# # #             if cost[r][c] < 60:
# # #                 self.tracks[r].update(boxes[c])
# # #                 matched_tracks.add(r)
# # #                 matched_det.add(c)

# # #         # new tracks
# # #         for i,b in enumerate(boxes):
# # #             if i not in matched_det:
# # #                 self.tracks.append(Track(b, self.next_id))
# # #                 self.next_id += 1

# # #         # remove old
# # #         self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
# # #         return self.tracks

# # # # ---------------------------------------------------------
# # # # FEATURE EXTRACTION
# # # # ---------------------------------------------------------
# # # def extract_features(track):
# # #     hist = list(track.history)
# # #     if len(hist) < 4:
# # #         return None

# # #     pts = np.array([(h[0],h[1]) for h in hist])
# # #     vel = np.linalg.norm(np.diff(pts,axis=0),axis=1)

# # #     mean_speed = np.mean(vel)
# # #     var_speed  = np.var(vel)
# # #     smoothness = np.mean(np.abs(np.diff(vel))) if len(vel)>=2 else 0
# # #     straightness = np.linalg.norm(pts[-1]-pts[0]) / (np.sum(vel)+1e-5)

# # #     sizes = np.array([h[2]*h[3] for h in hist])
# # #     mean_size = np.mean(sizes)
# # #     var_size  = np.var(sizes)

# # #     return np.array([mean_speed,var_speed,smoothness,mean_size,var_size,straightness])

# # # # ---------------------------------------------------------
# # # # RULE-BASED BACKUP DETECTOR
# # # # ---------------------------------------------------------
# # # def is_drone_rule(hist):
# # #     if len(hist) < 5:
# # #         return False
# # #     pts = np.array(hist)
# # #     vel = np.linalg.norm(np.diff(pts,axis=0),axis=1)
# # #     if np.var(vel) < 4 and np.mean(vel)>0.5:
# # #         return True
# # #     return False

# # # # ---------------------------------------------------------
# # # # CLASSIFIER
# # # # ---------------------------------------------------------
# # # class Classifier:
# # #     def __init__(self):
# # #         self.scaler = StandardScaler()
# # #         self.clf = None

# # #     def train(self, X, y):
# # #         Xs = self.scaler.fit_transform(X)
# # #         self.clf = SVC(kernel='rbf',probability=True)
# # #         self.clf.fit(Xs,y)

# # #     def predict(self, feat):
# # #         Xs = self.scaler.transform([feat])
# # #         return self.clf.predict(Xs)[0], float(np.max(self.clf.predict_proba(Xs)))

# # #     def save(self):
# # #         joblib.dump({"clf":self.clf,"scaler":self.scaler}, CONFIG["MODEL_OUT"])

# # #     def load(self):
# # #         d = joblib.load(CONFIG["MODEL_OUT"])
# # #         self.clf = d["clf"]
# # #         self.scaler = d["scaler"]

# # # # ---------------------------------------------------------
# # # # ANNOTATION FEATURE EXTRACTION
# # # # ---------------------------------------------------------
# # # def load_annotations():
# # #     if not os.path.exists(CONFIG["ANNOTATIONS_CSV"]):
# # #         print("[Annotation] No annotations.csv found → creating empty template.")
# # #         with open(CONFIG["ANNOTATIONS_CSV"],"w") as f:
# # #             f.write("video,frame_idx,x,y,w,h,label\n")
# # #         return np.empty((0,6)), []

# # #     X=[]; y=[]
# # #     with open(CONFIG["ANNOTATIONS_CSV"]) as f:
# # #         rdr=csv.reader(f)
# # #         next(rdr,None)
# # #         rows=list(rdr)

# # #     if len(rows)<10:
# # #         print("[Annotation] Not enough entries (<10). Skipping training.")
# # #         return np.empty((0,6)), []

# # #     # group by label
# # #     rows = sorted(rows,key=lambda r:(r[0],int(r[1])))

# # #     current=[]
# # #     for r in rows:
# # #         vid, fidx,x,y,w,h,label = r
# # #         current.append((int(x)+int(w)/2, int(y)+int(h)/2, int(w),int(h)))

# # #     # build track-like object
# # #     class Fake:
# # #         def __init__(self,h): self.history=h

# # #     track = Fake(current)
# # #     feat = extract_features(track)
# # #     if feat is not None:
# # #         X.append(feat); y.append(label)

# # #     return np.array(X), y

# # # # ---------------------------------------------------------
# # # # RUN DETECTOR ON VIDEO
# # # # ---------------------------------------------------------
# # # def run_detector(video_path, classifier):

# # #     cap = cv2.VideoCapture(video_path)
# # #     md = MotionDetector()
# # #     tr = TrackerManager()

# # #     outpath = os.path.join(CONFIG["OUTPUT_FOLDER"], 
# # #                            os.path.basename(video_path).split(".")[0] + "_output.mp4")

# # #     w = int(cap.get(3))
# # #     h = int(cap.get(4))
# # #     out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w,h))

# # #     frame_idx=0
# # #     print(f"[Run] Processing {video_path} ...")

# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         boxes,_ = md.get_candidates(frame)
# # #         tracks = tr.update(boxes)

# # #         for t in tracks:
# # #             feat = extract_features(t)
# # #             label="unknown"; prob=0.0

# # #             if classifier.clf and feat is not None:
# # #                 label,prob = classifier.predict(feat)
# # #             else:
# # #                 hist = [(h[0],h[1]) for h in t.history]
# # #                 label = "drone" if is_drone_rule(hist) else "bird"

# # #             x,y,w2,h2 = t.bbox
# # #             color = (0,0,255) if label=="drone" else (0,255,0)
# # #             cv2.rectangle(frame,(x,y),(x+w2,y+h2),color,2)
# # #             cv2.putText(frame,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

# # #         out.write(frame)
# # #         frame_idx+=1

# # #     cap.release()
# # #     out.release()
# # #     print(f"[Run] Saved output to {outpath}")

# # # # ---------------------------------------------------------
# # # # MAIN (AUTOMATIC WORKFLOW)
# # # # ---------------------------------------------------------
# # # if __name__ == "__main__":
# # #     print("\n=== FULLY AUTOMATIC DRONE-BIRD PIPELINE STARTED ===\n")

# # #     # 1) Extract frames from ALL videos
# # #     videos = [os.path.join(CONFIG["VIDEO_FOLDER"],v) 
# # #               for v in os.listdir(CONFIG["VIDEO_FOLDER"]) 
# # #               if v.lower().endswith((".mp4",".mov",".avi",".mkv"))]

# # #     if len(videos)==0:
# # #         print("❌ No videos in data/videos/. Add at least one video.")
# # #         exit()

# # #     print("\n[Step 1] Extracting frames ...")
# # #     for v in videos:
# # #         name = os.path.basename(v).split(".")[0]
# # #         out_folder = os.path.join(CONFIG["FRAMES_FOLDER"], name)
# # #         extract_frames(v, out_folder, CONFIG["FRAME_STEP"])

# # #     # 2) Load annotations
# # #     print("\n[Step 2] Loading annotations ...")
# # #     X,y = load_annotations()

# # #     # 3) Train classifier if annotations exist
# # #     clf = Classifier()
# # #     trained=False

# # #     if len(y)>=10:
# # #         print(f"[Step 3] Training SVM on {len(y)} samples ...")
# # #         clf.train(X,y)
# # #         clf.save()
# # #         trained=True
# # #         print("[Step 3] Model saved.")

# # #     # 4) Load model if exists
# # #     if os.path.exists(CONFIG["MODEL_OUT"]) and not trained:
# # #         print("[Step 3] Loading existing model ...")
# # #         clf.load()
# # #         trained=True

# # #     if not trained:
# # #         print("[Step 3] No model → using rule-based fallback")

# # #     # 5) Run detector on each video
# # #     print("\n[Step 4] Running detector ...")
# # #     for v in videos:
# # #         run_detector(v, clf)

# # #     print("\n=== PIPELINE COMPLETE ===\n")
# # # # #!/usr/bin/env python3
# # # # """
# # # # drone_bird_pipeline.py

# # # # Full end-to-end pipeline implementing:
# # # # - Option 2: modular motion-based tracker + temporal/shape features + SVM classifier
# # # # - Option 3: simple rule-based detector based on motion stability

# # # # Author: ChatGPT (GPT-5 Thinking mini)
# # # # Date: 2025-11-23

# # # # Usage:
# # # #     - Edit CONFIG below to point to your files/folders.
# # # #     - If you have annotations.csv, you can train an SVM using extract_features_from_annotations()
# # # #     - Run `python drone_bird_pipeline.py` to run the demo pipeline on a sample video.

# # # # """

# # # # import os
# # # # import cv2
# # # # import numpy as np
# # # # from collections import deque, defaultdict
# # # # from sklearn.svm import SVC
# # # # from sklearn.preprocessing import StandardScaler
# # # # import joblib
# # # # from filterpy.kalman import KalmanFilter
# # # # from scipy.optimize import linear_sum_assignment
# # # # import csv
# # # # import math
# # # # import time

# # # # # ---------------------------
# # # # # Configuration
# # # # # ---------------------------
# # # # CONFIG = {
# # # #     "VIDEO_FOLDER": "data/videos",   # folder with .mp4 files (or point to single file)
# # # #     "FRAMES_FOLDER": "data/frames",  # optional, if frames pre-extracted; else frames will be read from video
# # # #     "ANNOTATIONS_CSV": "data/annotations.csv",  # optional; if present, use for training
# # # #     "MODEL_OUT": "models/svm_drone_bird.pkl",
# # # #     "MIN_AREA": 10,
# # # #     "MAX_AREA": 5000,
# # # #     "TRACK_MAX_AGE": 8,
# # # #     "TRACK_MIN_HITS": 3,
# # # #     "FEATURE_HISTORY": 12,  # how many positions per track we keep for feature extraction
# # # # }

# # # # os.makedirs(os.path.dirname(CONFIG["MODEL_OUT"]), exist_ok=True)

# # # # # ---------------------------
# # # # # Utility functions
# # # # # ---------------------------
# # # # def ensure_dir(path):
# # # #     if path and not os.path.exists(path):
# # # #         os.makedirs(path)

# # # # def centroid_from_box(box):
# # # #     x, y, w, h = box
# # # #     return (int(x + w/2), int(y + h/2))

# # # # # ---------------------------
# # # # # Frame extraction helper (if needed)
# # # # # ---------------------------
# # # # def extract_frames_from_video(video_path, out_folder, step=5):
# # # #     """Extract frames every `step` frames and write to out_folder."""
# # # #     ensure_dir(out_folder)
# # # #     cam = cv2.VideoCapture(video_path)
# # # #     frame_idx = 0
# # # #     saved = 0
# # # #     while True:
# # # #         ret, frame = cam.read()
# # # #         if not ret:
# # # #             break
# # # #         if frame_idx % step == 0:
# # # #             fn = os.path.join(out_folder, f"frame_{frame_idx}.jpg")
# # # #             cv2.imwrite(fn, frame)
# # # #             saved += 1
# # # #         frame_idx += 1
# # # #     cam.release()
# # # #     print(f"[extract_frames_from_video] Saved {saved} frames to {out_folder}")

# # # # # ---------------------------
# # # # # Motion detection
# # # # # ---------------------------
# # # # class MotionDetector:
# # # #     def __init__(self):
# # # #         self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

# # # #     def get_mask(self, frame):
# # # #         """Return a clean binary motion mask."""
# # # #         fg = self.bg.apply(frame)
# # # #         fg = cv2.medianBlur(fg, 5)
# # # #         _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
# # # #         # morphological clean
# # # #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# # # #         fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
# # # #         fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
# # # #         return fg

# # # #     def get_candidates(self, frame, min_area=CONFIG["MIN_AREA"], max_area=CONFIG["MAX_AREA"]):
# # # #         """Return list of bounding boxes from motion mask."""
# # # #         mask = self.get_mask(frame)
# # # #         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # #         boxes = []
# # # #         for c in cnts:
# # # #             area = cv2.contourArea(c)
# # # #             if area < min_area or area > max_area:
# # # #                 continue
# # # #             x, y, w, h = cv2.boundingRect(c)
# # # #             boxes.append((x, y, w, h))
# # # #         # sort by size descending
# # # #         boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
# # # #         return boxes, mask

# # # # # ---------------------------
# # # # # Track class and manager (Kalman + assignment)
# # # # # ---------------------------
# # # # class Track:
# # # #     def __init__(self, bbox, track_id):
# # # #         cx, cy = centroid_from_box(bbox)
# # # #         self.id = track_id
# # # #         self.kf = self._init_kf(cx, cy)
# # # #         self.bbox = bbox
# # # #         self.hits = 1
# # # #         self.age = 0
# # # #         self.time_since_update = 0
# # # #         self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])  # store (cx,cy,w,h)
# # # #         self.history.append((cx, cy, bbox[2], bbox[3]))

# # # #     def _init_kf(self, cx, cy):
# # # #         kf = KalmanFilter(dim_x=4, dim_z=2)
# # # #         kf.F = np.array([[1,0,1,0],
# # # #                          [0,1,0,1],
# # # #                          [0,0,1,0],
# # # #                          [0,0,0,1]], dtype=float)
# # # #         kf.H = np.array([[1,0,0,0],
# # # #                          [0,1,0,0]], dtype=float)
# # # #         kf.x = np.array([cx, cy, 0, 0], dtype=float)
# # # #         kf.P *= 500.
# # # #         kf.R *= 5.
# # # #         kf.Q = np.eye(4) * 0.01
# # # #         return kf

# # # #     def predict(self):
# # # #         self.kf.predict()
# # # #         self.age += 1
# # # #         self.time_since_update += 1
# # # #         x, y = int(self.kf.x[0]), int(self.kf.x[1])
# # # #         return x, y

# # # #     def update(self, bbox):
# # # #         """bbox = (x,y,w,h)"""
# # # #         cx, cy = centroid_from_box(bbox)
# # # #         self.kf.update(np.array([cx, cy]))
# # # #         self.bbox = bbox
# # # #         self.history.append((cx, cy, bbox[2], bbox[3]))
# # # #         self.hits += 1
# # # #         self.time_since_update = 0

# # # #     def get_state(self):
# # # #         return (int(self.kf.x[0]), int(self.kf.x[1]))

# # # # class TrackerManager:
# # # #     def __init__(self):
# # # #         self.tracks = []
# # # #         self.next_id = 0

# # # #     def _associate(self, detections):
# # # #         """
# # # #         Associate detections to existing tracks using Euclidean distance and Hungarian.
# # # #         detections: list of centroid tuples (cx,cy)
# # # #         returns: matches (track_idx, det_idx), unmatched_tracks, unmatched_dets
# # # #         """
# # # #         if len(self.tracks) == 0:
# # # #             return [], list(range(len(self.tracks))), list(range(len(detections)))

# # # #         track_centers = [t.get_state() for t in self.tracks]
# # # #         if len(track_centers) == 0 or len(detections) == 0:
# # # #             return [], list(range(len(self.tracks))), list(range(len(detections)))

# # # #         cost = np.zeros((len(track_centers), len(detections)), dtype=float)
# # # #         for i, tc in enumerate(track_centers):
# # # #             for j, dc in enumerate(detections):
# # # #                 cost[i,j] = math.hypot(tc[0]-dc[0], tc[1]-dc[1])

# # # #         row_ind, col_ind = linear_sum_assignment(cost)
# # # #         matches, unmatched_tracks, unmatched_dets = [], [], []
# # # #         matched_tracks = set()
# # # #         matched_dets = set()
# # # #         for r,c in zip(row_ind, col_ind):
# # # #             if cost[r,c] > 60.0:  # threshold - too far => don't match
# # # #                 continue
# # # #             matches.append((r,c))
# # # #             matched_tracks.add(r); matched_dets.add(c)

# # # #         for t_i in range(len(self.tracks)):
# # # #             if t_i not in matched_tracks:
# # # #                 unmatched_tracks.append(t_i)
# # # #         for d_i in range(len(detections)):
# # # #             if d_i not in matched_dets:
# # # #                 unmatched_dets.append(d_i)

# # # #         return matches, unmatched_tracks, unmatched_dets

# # # #     def update(self, det_boxes):
# # # #         """
# # # #         det_boxes: list of (x,y,w,h)
# # # #         returns: list of active tracks after update
# # # #         """
# # # #         det_centroids = [centroid_from_box(b) for b in det_boxes]
# # # #         # predict all tracks
# # # #         for t in self.tracks:
# # # #             t.predict()

# # # #         matches, unmatched_tracks, unmatched_dets = self._associate(det_centroids)

# # # #         # update matched
# # # #         for trk_idx, det_idx in matches:
# # # #             self.tracks[trk_idx].update(det_boxes[det_idx])

# # # #         # create new tracks for unmatched detections
# # # #         for det_idx in unmatched_dets:
# # # #             new_trk = Track(det_boxes[det_idx], self.next_id)
# # # #             self.next_id += 1
# # # #             self.tracks.append(new_trk)

# # # #         # increase age; remove dead tracks
# # # #         to_remove = []
# # # #         for i, t in enumerate(self.tracks):
# # # #             if t.time_since_update > CONFIG["TRACK_MAX_AGE"]:
# # # #                 to_remove.append(i)
# # # #         # remove in reverse order
# # # #         for i in reversed(to_remove):
# # # #             del self.tracks[i]

# # # #         return self.tracks

# # # # # ---------------------------
# # # # # Feature extraction (Approach A + C)
# # # # # ---------------------------
# # # # def extract_features_from_track(track):
# # # #     """
# # # #     Input: Track object with history deque of (cx,cy,w,h)
# # # #     Output: feature vector (numpy array)
# # # #     Features included:
# # # #       - mean speed
# # # #       - speed variance
# # # #       - smoothness (mean change in velocity)
# # # #       - mean bbox size, size variance
# # # #       - edge_density_stability (if images available, optional)
# # # #     """
# # # #     hist = list(track.history)
# # # #     if len(hist) < 4:
# # # #         return None
# # # #     pts = np.array([(h[0], h[1]) for h in hist], dtype=float)
# # # #     sizes = np.array([h[2]*h[3] for h in hist], dtype=float)

# # # #     # velocities between consecutive centers
# # # #     vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
# # # #     mean_speed = float(np.mean(vel))
# # # #     var_speed = float(np.var(vel))
# # # #     # smoothness: mean magnitude of acceleration (changes in velocities)
# # # #     if len(vel) >= 2:
# # # #         smoothness = float(np.mean(np.abs(np.diff(vel))))
# # # #     else:
# # # #         smoothness = 0.0

# # # #     mean_size = float(np.mean(sizes))
# # # #     var_size = float(np.var(sizes))

# # # #     # trajectory straightness: ratio distance(ends)/sum of segment lengths
# # # #     path_len = float(np.sum(vel))
# # # #     end2end = float(np.linalg.norm(pts[-1] - pts[0]))
# # # #     straightness = end2end / (path_len + 1e-6)

# # # #     feats = np.array([mean_speed, var_speed, smoothness, mean_size, var_size, straightness], dtype=float)
# # # #     return feats

# # # # # ---------------------------
# # # # # Classifier wrapper (SVM)
# # # # # ---------------------------
# # # # class DroneBirdClassifier:
# # # #     def __init__(self):
# # # #         self.clf = None
# # # #         self.scaler = StandardScaler()

# # # #     def train(self, X, y):
# # # #         """
# # # #         X: np.array (N x D)
# # # #         y: list/array of labels (strings 'drone'/'bird')
# # # #         """
# # # #         if len(X) == 0:
# # # #             raise ValueError("No training samples provided")
# # # #         Xs = self.scaler.fit_transform(X)
# # # #         self.clf = SVC(kernel='rbf', probability=True)
# # # #         self.clf.fit(Xs, y)
# # # #         return self

# # # #     def predict(self, feats):
# # # #         if self.clf is None:
# # # #             return "unknown"
# # # #         if feats is None:
# # # #             return "unknown"
# # # #         Xs = self.scaler.transform([feats])
# # # #         p = self.clf.predict(Xs)[0]
# # # #         prob = np.max(self.clf.predict_proba(Xs))
# # # #         return p, float(prob)

# # # #     def save(self, path):
# # # #         joblib.dump({"clf": self.clf, "scaler": self.scaler}, path)

# # # #     def load(self, path):
# # # #         d = joblib.load(path)
# # # #         self.clf = d["clf"]
# # # #         self.scaler = d["scaler"]
# # # #         return self

# # # # # ---------------------------
# # # # # Annotation-based feature extraction (training helper)
# # # # # ---------------------------
# # # # def extract_features_from_annotations(annotations_csv, frames_root=None):
# # # #     """
# # # #     Parse annotations CSV and convert to track-level features.
# # # #     It expects contiguous frame entries for same object label per track.
# # # #     This function returns X (features) and y (labels).
# # # #     If frames_root is provided, it could compute image-based features (not used here).
# # # #     """
# # # #     # Read CSV rows and group by track key (video, track_id)
# # # #     # We expect user to provide contiguous bounding boxes for the same track.
# # # #     rows = []
# # # #     with open(annotations_csv, newline='') as f:
# # # #         rdr = csv.reader(f)
# # # #         for r in rdr:
# # # #             if len(r) < 7:
# # # #                 continue
# # # #             video, frame_idx, x, y, w, h, label = r[:7]
# # # #             rows.append((video, int(frame_idx), int(x), int(y), int(w), int(h), label.strip()))
# # # #     # Group by video and label sequence; naive grouping by contiguous frames
# # # #     rows = sorted(rows, key=lambda r: (r[0], r[1]))
# # # #     tracks = []
# # # #     current = None
# # # #     for r in rows:
# # # #         vid, fidx, x, y, w, h, label = r
# # # #         if current is None:
# # # #             current = {"video": vid, "label": label, "frames": [(fidx, (x,y,w,h))]}
# # # #             last_frame = fidx
# # # #         else:
# # # #             if r[0] == current["video"] and label == current["label"] and fidx - last_frame <= 10:
# # # #                 current["frames"].append((fidx, (x,y,w,h)))
# # # #                 last_frame = fidx
# # # #             else:
# # # #                 tracks.append(current)
# # # #                 current = {"video": vid, "label": label, "frames": [(fidx, (x,y,w,h))]}
# # # #                 last_frame = fidx
# # # #     if current:
# # # #         tracks.append(current)

# # # #     # Now compute features per track
# # # #     X, y = [], []
# # # #     for t in tracks:
# # # #         if len(t["frames"]) < 4:
# # # #             continue
# # # #         # build a fake Track-like history to use extract_features_from_track
# # # #         class _FakeTrack:
# # # #             def __init__(self, entries):
# # # #                 self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])
# # # #                 for (_, (x,y,w,h)) in entries:
# # # #                     cx, cy = int(x + w/2), int(y + h/2)
# # # #                     self.history.append((cx, cy, w, h))
# # # #         ft = _FakeTrack(t["frames"])
# # # #         feats = extract_features_from_track(ft)
# # # #         if feats is not None:
# # # #             X.append(feats)
# # # #             y.append(t["label"])
# # # #     X = np.array(X) if len(X) else np.empty((0,6))
# # # #     return X, y

# # # # # ---------------------------
# # # # # Minimal rule-based detector (Option 3)
# # # # # ---------------------------
# # # # def is_drone_rule_based(track_history):
# # # #     """
# # # #     rule-based heuristic:
# # # #       - requires >=5 points
# # # #       - low variance in speed -> likely drone
# # # #       - high straightness -> likely drone
# # # #     """
# # # #     if len(track_history) < 5:
# # # #         return False
# # # #     pts = np.array(track_history, dtype=float)
# # # #     vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
# # # #     var_speed = np.var(vel)
# # # #     mean_speed = np.mean(vel)
# # # #     path_len = np.sum(vel)
# # # #     end2end = np.linalg.norm(pts[-1] - pts[0])
# # # #     straightness = end2end / (path_len + 1e-6)
# # # #     # thresholds (tunable)
# # # #     if var_speed < 4.0 and straightness > 0.6 and mean_speed > 0.5:
# # # #         return True
# # # #     return False

# # # # def minimal_detector_on_video(video_path):
# # # #     """Run the minimal rule-based detector over video and print alarms."""
# # # #     cap = cv2.VideoCapture(video_path)
# # # #     md = MotionDetector()
# # # #     track = []  # simple single-track history of centroids
# # # #     frame_idx = 0
# # # #     while True:
# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break
# # # #         boxes, mask = md.get_candidates(frame)
# # # #         if boxes:
# # # #             x,y,w,h = boxes[0]
# # # #             cx, cy = centroid_from_box((x,y,w,h))
# # # #             track.append((cx, cy))
# # # #             if is_drone_rule_based(track[-12:]):
# # # #                 print(f"[Frame {frame_idx}] DRONE (rule-based) at {cx,cy}")
# # # #                 # draw and show small annotation (optional)
# # # #                 cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),1)
# # # #                 cv2.putText(frame, "DRONE (rule)", (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
# # # #         else:
# # # #             # no detections: keep history but don't append
# # # #             pass

# # # #         frame_idx += 1

# # # #     cap.release()
# # # #     print("Completed minimal run.")

# # # # # ---------------------------
# # # # # Demo / main
# # # # # ---------------------------
# # # # def demo_run_on_video(video_path, classifier=None, save_output=None, visualize=False):
# # # #     """
# # # #     Run full pipeline on video. If classifier provided, use it for decisions,
# # # #     otherwise fall back to the minimal rule-based decision.
# # # #     """
# # # #     cap = cv2.VideoCapture(video_path)
# # # #     md = MotionDetector()
# # # #     tracker_mgr = TrackerManager()
# # # #     frame_idx = 0

# # # #     out_writer = None
# # # #     if save_output is not None:
# # # #         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# # # #         out_writer = None

# # # #     while True:
# # # #         ret, frame = cap.read()
# # # #         if not ret:
# # # #             break

# # # #         det_boxes, mask = md.get_candidates(frame)
# # # #         tracks = tracker_mgr.update(det_boxes)

# # # #         # for each active track compute features and decide
# # # #         for t in tracks:
# # # #             feats = extract_features_from_track(t)
# # # #             label = "unknown"
# # # #             prob = 0.0
# # # #             if classifier is not None and feats is not None:
# # # #                 pred = classifier.predict(feats)
# # # #                 if isinstance(pred, tuple):
# # # #                     label, prob = pred
# # # #                 else:
# # # #                     label = pred
# # # #             else:
# # # #                 # fallback to rule-based on this track history
# # # #                 hist_pts = [(h[0], h[1]) for h in t.history]
# # # #                 if is_drone_rule_based(hist_pts):
# # # #                     label = "drone"
# # # #                 else:
# # # #                     label = "bird_or_unknown"

# # # #             # show on frame
# # # #             cx, cy = t.get_state()
# # # #             x,y,w,h = t.bbox
# # # #             color = (0,255,0) if label=="bird_or_unknown" else (0,0,255)
# # # #             cv2.rectangle(frame, (x,y),(x+w,y+h), color, 1)
# # # #             cv2.putText(frame, f"{label[:6]} {prob:.2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# # # #             # if drone, print alarm
# # # #             if label == "drone":
# # # #                 print(f"[Frame {frame_idx}] ALARM: DRONE detected track {t.id} at ({cx},{cy}) prob={prob:.2f}")

# # # #         if visualize:
# # # #             cv2.imshow("frame", frame)
# # # #             key = cv2.waitKey(1)
# # # #             if key == 27:
# # # #                 break

# # # #         frame_idx += 1

# # # #     cap.release()
# # # #     if visualize:
# # # #         cv2.destroyAllWindows()

# # # # # ---------------------------
# # # # # Example usage when run as script
# # # # # ---------------------------
# # # # if __name__ == "__main__":
# # # #     # Simple CLI-like flow
# # # #     # 1) If annotations available, extract features and train SVM
# # # #     # 2) Run demo pipeline on first video found

# # # #     # TRAINING (if annotations provided)
# # # #     ann = CONFIG["ANNOTATIONS_CSV"]
# # # #     classifier = DroneBirdClassifier()
# # # #     trained = False
# # # #     if os.path.exists(ann):
# # # #         print("[Main] Found annotations, extracting features...")
# # # #         X, y = extract_features_from_annotations(ann)
# # # #         print(f"[Main] Got {len(y)} labeled tracks (features).")
# # # #         if len(y) >= 10:
# # # #             classifier.train(X, y)
# # # #             classifier.save(CONFIG["MODEL_OUT"])
# # # #             trained = True
# # # #             print("[Main] Trained and saved model ->", CONFIG["MODEL_OUT"])
# # # #         else:
# # # #             print("[Main] Not enough labeled tracks to train robust SVM (need >=10).")

# # # #     # If a saved model exists, load it
# # # #     if os.path.exists(CONFIG["MODEL_OUT"]) and not trained:
# # # #         try:
# # # #             classifier.load(CONFIG["MODEL_OUT"])
# # # #             trained = True
# # # #             print("[Main] Loaded pretrained model")
# # # #         except Exception as e:
# # # #             print("Failed loading model:", e)

# # # #     # Demo run: choose a video from VIDEO_FOLDER
# # # #     video_list = []
# # # #     if os.path.isdir(CONFIG["VIDEO_FOLDER"]):
# # # #         for fn in os.listdir(CONFIG["VIDEO_FOLDER"]):
# # # #             if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
# # # #                 video_list.append(os.path.join(CONFIG["VIDEO_FOLDER"], fn))
# # # #     if len(video_list) == 0:
# # # #         print("[Main] No videos found in VIDEO_FOLDER. Exiting. You can point CONFIG['VIDEO_FOLDER'] to a video file.")
# # # #     else:
# # # #         demo_video = video_list[0]
# # # #         print("[Main] Running demo on:", demo_video)
# # # #         demo_run_on_video(demo_video, classifier=(classifier if trained else None), visualize=False)

# # # #     # Also provide an option to quickly run the minimal detector:
# # # #     # minimal_detector_on_video(demo_video)
# # #!/usr/bin/env python3
# # """
# # FULLY AUTOMATIC DRONE VS BIRD PIPELINE
# # --------------------------------------

# # What this script does AUTOMATICALLY:

# # 1. Creates required folders:
# #        data/videos/
# #        data/frames/
# #        data/annotations.csv (template)
# #        models/
# #        output/

# # 2. Extracts frames ONLY IF NOT ALREADY EXTRACTED

# # 3. Trains SVM if annotations.csv has >= 10 samples

# # 4. Loads model if exists, else uses rule-based detector

# # 5. Runs UAV-vs-Bird detection on every video
# #    and saves annotated output video into output/

# # No user action required.
# # """

# # import os
# # import cv2
# # import numpy as np
# # from collections import deque
# # from sklearn.svm import SVC
# # from sklearn.preprocessing import StandardScaler
# # import joblib
# # from filterpy.kalman import KalmanFilter
# # from scipy.optimize import linear_sum_assignment
# # import csv
# # import math

# # # ---------------------------------------------------------
# # # CONFIGURATION
# # # ---------------------------------------------------------
# # CONFIG = {
# #     "VIDEO_FOLDER": "data/videos",
# #     "FRAMES_FOLDER": "data/frames",
# #     "ANNOTATIONS_CSV": "data/annotations.csv",
# #     "MODEL_OUT": "models/svm_drone_bird.pkl",
# #     "OUTPUT_FOLDER": "output",
# #     "FRAME_STEP": 5,
# #     "MIN_AREA": 10,
# #     "MAX_AREA": 5000,
# #     "FEATURE_HISTORY": 12,
# #     "TRACK_MAX_AGE": 8,
# # }

# # # ---------------------------------------------------------
# # # Ensure all folders exist
# # # ---------------------------------------------------------
# # for folder in [
# #     CONFIG["VIDEO_FOLDER"], CONFIG["FRAMES_FOLDER"],
# #     os.path.dirname(CONFIG["MODEL_OUT"]), CONFIG["OUTPUT_FOLDER"]
# # ]:
# #     os.makedirs(folder, exist_ok=True)

# # # ---------------------------------------------------------
# # # Centroid utility
# # # ---------------------------------------------------------
# # def centroid_from_box(b):
# #     x, y, w, h = b
# #     return int(x + w/2), int(y + h/2)

# # # ---------------------------------------------------------
# # # FRAME EXTRACTION WITH SKIP LOGIC
# # # ---------------------------------------------------------
# # def extract_frames(video_path, out_folder, step=5):
# #     os.makedirs(out_folder, exist_ok=True)
# #     cap = cv2.VideoCapture(video_path)
# #     idx = 0
# #     saved = 0

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         if idx % step == 0:
# #             cv2.imwrite(os.path.join(out_folder, f"frame_{idx}.jpg"), frame)
# #             saved += 1
# #         idx += 1

# #     cap.release()
# #     print(f"[Frames] Saved {saved} frames → {out_folder}")

# # # ---------------------------------------------------------
# # # MOTION DETECTOR
# # # ---------------------------------------------------------
# # class MotionDetector:
# #     def __init__(self):
# #         self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16)

# #     def get_candidates(self, frame):
# #         mask = self.bg.apply(frame)
# #         mask = cv2.medianBlur(mask, 5)
# #         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# #         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         boxes = []

# #         for c in cnts:
# #             area = cv2.contourArea(c)
# #             if CONFIG["MIN_AREA"] < area < CONFIG["MAX_AREA"]:
# #                 x, y, w, h = cv2.boundingRect(c)
# #                 boxes.append((x, y, w, h))

# #         return boxes, mask

# # # ---------------------------------------------------------
# # # KALMAN TRACKING
# # # ---------------------------------------------------------
# # class Track:
# #     def __init__(self, box, tid):
# #         cx, cy = centroid_from_box(box)
# #         self.id = tid
# #         self.bbox = box
# #         self.history = deque(maxlen=CONFIG["FEATURE_HISTORY"])
# #         self.history.append((cx, cy, box[2], box[3]))

# #         self.kf = self.init_kf(cx, cy)
# #         self.time_since_update = 0

# #     def init_kf(self, cx, cy):
# #         kf = KalmanFilter(dim_x=4, dim_z=2)
# #         kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], float)
# #         kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
# #         kf.x = np.array([cx, cy, 0, 0], float)
# #         kf.P *= 500
# #         kf.R *= 5
# #         kf.Q = np.eye(4) * 0.01
# #         return kf

# #     def predict(self):
# #         self.kf.predict()
# #         self.time_since_update += 1

# #     def update(self, box):
# #         cx, cy = centroid_from_box(box)
# #         self.kf.update(np.array([cx, cy]))
# #         self.bbox = box
# #         self.history.append((cx, cy, box[2], box[3]))
# #         self.time_since_update = 0

# #     def pos(self):
# #         return int(self.kf.x[0]), int(self.kf.x[1])

# # class TrackerManager:
# #     def __init__(self):
# #         self.tracks = []
# #         self.next_id = 0

# #     def update(self, boxes):
# #         # Predict step for all tracks
# #         for t in self.tracks:
# #             t.predict()

# #         if len(boxes) == 0:
# #             return self.tracks

# #         det_centers = [centroid_from_box(b) for b in boxes]
# #         track_centers = [t.pos() for t in self.tracks]

# #         # Cost matrix
# #         cost = np.zeros((len(track_centers), len(det_centers)))
# #         for i, tc in enumerate(track_centers):
# #             for j, dc in enumerate(det_centers):
# #                 cost[i, j] = math.dist(tc, dc)

# #         # Assign using Hungarian
# #         row, col = linear_sum_assignment(cost)
# #         matched_tracks = set()
# #         matched_det = set()

# #         for r, c in zip(row, col):
# #             if cost[r, c] < 60:
# #                 self.tracks[r].update(boxes[c])
# #                 matched_tracks.add(r)
# #                 matched_det.add(c)

# #         # Create new tracks for unmatched detections
# #         for i, b in enumerate(boxes):
# #             if i not in matched_det:
# #                 self.tracks.append(Track(b, self.next_id))
# #                 self.next_id += 1

# #         # Remove stale tracks
# #         self.tracks = [t for t in self.tracks if t.time_since_update <= CONFIG["TRACK_MAX_AGE"]]
# #         return self.tracks

# # # ---------------------------------------------------------
# # # FEATURE EXTRACTION
# # # ---------------------------------------------------------
# # def extract_features(track):
# #     hist = list(track.history)
# #     if len(hist) < 4:
# #         return None

# #     pts = np.array([(h[0], h[1]) for h in hist])
# #     vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)

# #     mean_speed = vel.mean()
# #     var_speed = vel.var()
# #     smoothness = np.abs(np.diff(vel)).mean() if len(vel) >= 2 else 0
# #     straight = np.linalg.norm(pts[-1] - pts[0]) / (vel.sum() + 1e-5)

# #     sizes = np.array([h[2] * h[3] for h in hist])
# #     return np.array([mean_speed, var_speed, smoothness, sizes.mean(), sizes.var(), straight])

# # # ---------------------------------------------------------
# # # RULE-BASED BACKUP
# # # ---------------------------------------------------------
# # def is_drone_rule(hist):
# #     if len(hist) < 5:
# #         return False
# #     pts = np.array(hist)
# #     vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
# #     return vel.mean() > 0.5 and vel.var() < 4

# # # ---------------------------------------------------------
# # # CLASSIFIER
# # # ---------------------------------------------------------
# # class Classifier:
# #     def __init__(self):
# #         self.clf = None
# #         self.scaler = StandardScaler()

# #     def train(self, X, y):
# #         Xs = self.scaler.fit_transform(X)
# #         self.clf = SVC(kernel="rbf", probability=True)
# #         self.clf.fit(Xs, y)

# #     def predict(self, feat):
# #         Xs = self.scaler.transform([feat])
# #         label = self.clf.predict(Xs)[0]
# #         prob = self.clf.predict_proba(Xs).max()
# #         return label, float(prob)

# #     def save(self):
# #         joblib.dump({"clf": self.clf, "scaler": self.scaler}, CONFIG["MODEL_OUT"])

# #     def load(self):
# #         d = joblib.load(CONFIG["MODEL_OUT"])
# #         self.clf = d["clf"]
# #         self.scaler = d["scaler"]

# # # ---------------------------------------------------------
# # # LOAD ANNOTATIONS
# # # ---------------------------------------------------------
# # # def load_annotations():
# # #     csv_path = CONFIG["ANNOTATIONS_CSV"]

# # #     if not os.path.exists(csv_path):
# # #         print("[Annotations] No annotations.csv → creating template.")
# # #         with open(csv_path, "w") as f:
# # #             f.write("video,frame_idx,x,y,w,h,label\n")
# # #         return np.empty((0, 6)), []

# # #     X, y = [], []
# # #     with open(csv_path) as f:
# # #         rdr = csv.reader(f)
# # #         next(rdr, None)
# # #         rows = list(rdr)

# # #     if len(rows) < 10:
# # #         print("[Annotations] <10 entries → skipping training.")
# # #         return np.empty((0, 6)), []

# # #     # create one "fake" long track from all annotations
# # #     hist = []
# # #     for r in rows:
# # #         _, _, x, yb, w, h, label = r
# # #         hist.append((int(x) + int(w)/2, int(yb) + int(h)/2, int(w), int(h)))

# # #     class Fake:
# # #         def __init__(self, h): self.history = h

# # #     feat = extract_features(Fake(hist))
# # #     if feat is not None:
# # #         X.append(feat)
# # #         y.append(rows[-1][6])  # last label

# # #     return np.array(X), y
# # def load_annotations():
# #     ann_folder = "annotations"

# #     if not os.path.exists(ann_folder):
# #         print("[Annotation] Folder 'annotations/' not found.")
# #         return np.empty((0, 6)), []

# #     X = []
# #     y = []

# #     txt_files = [f for f in os.listdir(ann_folder) if f.endswith(".txt")]
# #     if len(txt_files) == 0:
# #         print("[Annotation] No .txt files found.")
# #         return np.empty((0,6)), []

# #     for txt in txt_files:
# #         path = os.path.join(ann_folder, txt)
# #         tracks = {}  # track_id → list of (cx,cy,w,h)

# #         with open(path) as f:
# #             for line in f:
# #                 parts = line.strip().split()

# #                 # skip invalid lines
# #                 if len(parts) != 7:
# #                     continue

# #                 frame, tid, x, yb, w, h, label = parts
# #                 frame = int(frame)
# #                 tid = int(tid)

# #                 # Convert to ints
# #                 x = int(x);  yb = int(yb)
# #                 w = int(w);  h = int(h)

# #                 # Compute center
# #                 cx = x + w/2
# #                 cy = yb + h/2

# #                 # Append to track
# #                 if tid not in tracks:
# #                     tracks[tid] = {"hist": [], "label": label}

# #                 tracks[tid]["hist"].append((cx, cy, w, h))

# #         # convert each track into feature vector
# #         for tid in tracks:
# #             hist = tracks[tid]["hist"]
# #             if len(hist) < 4:
# #                 continue

# #             class Fake:
# #                 def __init__(self, h):
# #                     self.history = h

# #             feats = extract_features(Fake(hist))
# #             if feats is not None:
# #                 X.append(feats)
# #                 y.append(tracks[tid]["label"])

# #     if len(y) == 0:
# #         print("[Annotation] No valid labeled tracks found.")
# #         return np.empty((0,6)), []

# #     print(f"[Annotation] Loaded {len(y)} labeled tracks from {len(txt_files)} annotation files.")
# #     return np.array(X), y

# # # ---------------------------------------------------------
# # # RUN DETECTOR ON VIDEO
# # # ---------------------------------------------------------
# # def run_detector(video_path, classifier):
# #     cap = cv2.VideoCapture(video_path)
# #     md = MotionDetector()
# #     tr = TrackerManager()

# #     out_path = os.path.join(CONFIG["OUTPUT_FOLDER"],
# #             os.path.basename(video_path).split(".")[0] + "_output.mp4")

# #     w, h = int(cap.get(3)), int(cap.get(4))
# #     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

# #     print(f"[Run] Running detection on: {video_path}")

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         boxes, _ = md.get_candidates(frame)
# #         tracks = tr.update(boxes)

# #         for t in tracks:
# #             feat = extract_features(t)
# #             if classifier.clf and feat is not None:
# #                 label, prob = classifier.predict(feat)
# #             else:
# #                 hist = [(h[0], h[1]) for h in t.history]
# #                 label = "drone" if is_drone_rule(hist) else "bird"
# #                 prob = 0.0

# #             x, yb, w2, h2 = t.bbox
# #             color = (0, 0, 255) if label == "drone" else (0, 255, 0)

# #             cv2.rectangle(frame, (x, yb), (x+w2, yb+h2), color, 2)
# #             cv2.putText(frame, label, (x, yb-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# #         out.write(frame)

# #     out.release()
# #     cap.release()

# #     print(f"[Run] Saved → {out_path}")

# # # ---------------------------------------------------------
# # # MAIN WORKFLOW
# # # ---------------------------------------------------------
# # if __name__ == "__main__":
# #     print("\n=== FULL AUTOMATIC DRONE-BIRD PIPELINE STARTED ===\n")

# #     # Load videos
# #     videos = [
# #         os.path.join(CONFIG["VIDEO_FOLDER"], f)
# #         for f in os.listdir(CONFIG["VIDEO_FOLDER"])
# #         if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
# #     ]

# #     if not videos:
# #         print("❌ No videos found in data/videos/")
# #         exit()

# #     # Step 1 — Frame extraction only if missing
# #     print("[Step 1] Checking frames...")
# #     for v in videos:
# #         name = os.path.basename(v).split(".")[0]
# #         out_folder = os.path.join(CONFIG["FRAMES_FOLDER"], name)

# #         if os.path.exists(out_folder) and os.listdir(out_folder):
# #             print(f"[Frames] Found existing frames for {name}. Skipping.")
# #         else:
# #             print(f"[Frames] No frames for {name}. Extracting...")
# #             extract_frames(v, out_folder, CONFIG["FRAME_STEP"])

# #     # Step 2 — Load annotations
# #     print("\n[Step 2] Loading annotations...")
# #     X, y = load_annotations()

# #     # Step 3 — Train or load model
# #     clf = Classifier()
# #     trained = False

# #     if len(y) >= 10:
# #         print(f"[Step 3] Training SVM on {len(y)} samples...")
# #         clf.train(X, y)
# #         clf.save()
# #         trained = True
# #         print("[Step 3] Model saved.")

# #     elif os.path.exists(CONFIG["MODEL_OUT"]):
# #         print("[Step 3] Loading saved model...")
# #         clf.load()
# #         trained = True

# #     else:
# #         print("[Step 3] No model → Using rule-based fallback")

# #     # Step 4 — Run detector
# #     print("\n[Step 4] Running detection on videos...\n")
# #     for v in videos:
# #         run_detector(v, clf)

# #     print("\n=== PIPELINE COMPLETE ===\n")
