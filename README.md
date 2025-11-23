#  Motion-Based Auto-Learned Rule Detector (Drone Classification)

This document summarizes the **Motion-Based Auto-Learned Rule Detector** approach (Option A) for classifying moving objects as either drones or other aerial objects (e.g., birds, noise) based purely on their motion signatures.

---

##  Approach Summary: Motion-Based Auto-Learned Rule Detector

This system processes video streams to detect, track, and classify objects using a combination of classical computer vision techniques and statistical auto-learning.

### 1. Motion Detection

* **Method:** Utilizes the **MOG2 (Mixture of Gaussians 2)** background subtraction algorithm to isolate foreground movement.
* **Noise Reduction:** Noise is filtered by removing detected blobs (moving regions) whose bounding box area falls outside a predefined minimum and maximum range.
* **Output:** Bounding boxes corresponding to all detected moving regions.

### 2. Multi-Object Tracking

Moving objects are persistently tracked over time to accumulate motion history.

* **Core Algorithm:** Each detected object is assigned a unique track and managed using a **Kalman Filter** for state estimation and prediction.
* **Data Association:** The **Hungarian matching algorithm** is used to reliably assign new detections to existing tracks, minimizing ID switches.
* **Track Management:** Logic is implemented for the creation of new tracks, updating existing ones, and removing tracks that have been inactive ("stale") for a defined period.
* **History Storage:** Each active track maintains a short history of its recent positions, speeds, and bounding box sizes.

### 3. Feature Extraction (Motion Signature)

For every active track, a set of numerical features are computed from its motion history. These features collectively form the object's **"motion signature"**:

* **Speed Metrics:** Mean speed, Speed variance, Smoothness (rate of change in speed).
* **Size Metrics:** Mean bounding-box size, Bounding-box size variance.
* **Trajectory Metrics:** **Straightness** of trajectory (deviation from a straight line).
* **Stability Metrics:** Bounding-box **jitter** (intensity of bounding box flapping/unstable motion).

### 4. Auto-Learning from Annotations

The core rule set is derived automatically from provided drone flight tracks (annotations).

* **Input:** Annotation files containing bounding box data for confirmed drone flights.
* **Feature Vector Generation:** The feature extraction process (Section 3) is applied to each confirmed drone track to generate its feature vector.
* **Threshold Computation:** Statistical metrics are computed across the entire set of drone feature vectors to define the **acceptable range** of "drone behavior":
    * Mean ($\mu$)
    * Standard Deviation ($\sigma$)
    * Percentile-based thresholds

### 5. Rule-Based Drone Classification

A new, unknown track is classified by comparing its calculated motion signature against the auto-learned drone behavior thresholds.

* **Classification Rule:** A track is classified as a drone if the majority of its computed feature values fall **inside** the learned acceptable ranges.
* **Reinforcement:** Features indicating high straightness and low jitter strongly reinforce a "drone" decision.
* **Final Decision:** Classification is made using **majority voting** across all motion features.

### 6. Output & Visualization

The system provides both visual and textual outputs for analysis and alerting.

* **Annotated Video:** An MP4 file is produced with bounding boxes drawn around detected objects:
    * **Red Bounding Box:** Object classified as a **drone**.
    * **Green Bounding Box:** Object classified as **other** (e.g., bird, noise, non-drone).
* **Alerts:** Textual alerts are printed to the console, detailing the object's position and the video frame index at which the classification occurred.

---

##  Drawbacks and Limitations (Concise)

The reliance solely on motion cues introduces several critical drawbacks:

* **Bird Misclassification:** **Straight, smooth bird trajectories** can easily be misclassified as drones, as the motion signature is identical.
* **Background Noise Failure:** The detector is highly sensitive to environmental factors. Strong background motion (e.g., wind in trees, water waves) produces **false blobs** that generate inaccurate tracks and features.
* **Generalization Failure:** The detector is **limited to the annotated behavior**. If real-world drone flight patterns deviate significantly from the patterns present in the training annotations, the learned thresholds will fail to classify new targets correctly.
* **Annotation Sensitivity:** The detector's reliability hinges on the **quality of the annotation data**. Incorrect or noisy bounding boxes in the training data directly corrupt the learned mean and threshold values.
* **No Appearance Information:** The complete lack of appearance (color, shape, texture) cues makes separating similarly moving objects (e.g., a small drone vs. a small, straight-flying bird) inherently difficult.
* **Tracking Failures:** The entire feature set relies on **stable track IDs**. If the tracking algorithm loses an object or prematurely switches IDs, the resulting motion features become unreliable or reset.
* **Small Object Noise:** Very small, long-range drones or small birds yield highly **noisy bounding box estimates**. This leads to unstable and unreliable motion features (speed, jitter, size variance).
