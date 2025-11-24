"""
Scratch-built alternative motion tracker that avoids NumPy/SciPy/SKLearn.

This module reimplements the numerical operations used by the original
`motion_tracker.py` using plain Python lists and lightweight helper
functions. It demonstrates how to build:
- Vector math utilities
- Local polynomial smoothing filters
- Manual finite-difference derivatives
- Basic K-Means clustering

OpenCV is still used for video I/O and image-level processing.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

Vector2 = Tuple[float, float]


# =============================================================================
# Low-level numerical helpers
# =============================================================================

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _vector_add(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] + b[0], a[1] + b[1])


def _vector_sub(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] - b[0], a[1] - b[1])


def _vector_scale(a: Vector2, scalar: float) -> Vector2:
    return (a[0] * scalar, a[1] * scalar)


def _vector_norm(a: Vector2) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    return math.sqrt(variance) or 1.0


def _moving_average(series: Sequence[float], window: int) -> List[float]:
    if window <= 1 or len(series) < 2:
        return list(series)

    half = window // 2
    smoothed: List[float] = []
    for idx in range(len(series)):
        start = max(0, idx - half)
        end = min(len(series), idx + half + 1)
        slice_vals = series[start:end]
        smoothed.append(_mean(slice_vals))
    return smoothed


def _solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """
    Simple Gaussian elimination with partial pivoting.
    """
    n = len(rhs)
    for i in range(n):
        # Pivot
        pivot_row = max(range(i, n), key=lambda r: abs(matrix[r][i]))
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
        rhs[i], rhs[pivot_row] = rhs[pivot_row], rhs[i]

        pivot = matrix[i][i] or 1e-12
        inv_pivot = 1.0 / pivot
        for j in range(i, n):
            matrix[i][j] *= inv_pivot
        rhs[i] *= inv_pivot

        for r in range(n):
            if r == i:
                continue
            factor = matrix[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n):
                matrix[r][c] -= factor * matrix[i][c]
            rhs[r] -= factor * rhs[i]

    return rhs


def _local_polynomial(series: Sequence[float], window: int, poly_order: int) -> List[float]:
    """
    Fit a polynomial around each sample using least squares and evaluate at center.
    """
    if len(series) < 2 or window < 3:
        return list(series)

    window = max(3, window | 1)  # force odd
    poly_order = max(1, min(poly_order, window - 1))
    half = window // 2
    result: List[float] = []

    for center in range(len(series)):
        xs: List[float] = []
        ys: List[float] = []

        for offset in range(-half, half + 1):
            idx = _clamp(center + offset, 0, len(series) - 1)
            xs.append(float(offset))
            ys.append(float(series[idx]))

        # Build normal equations A^T A * coeffs = A^T * y
        degree = poly_order + 1
        ata = [[0.0 for _ in range(degree)] for _ in range(degree)]
        aty = [0.0 for _ in range(degree)]

        for x_val, y_val in zip(xs, ys):
            basis = [x_val ** p for p in range(degree)]
            for r in range(degree):
                aty[r] += basis[r] * y_val
                for c in range(degree):
                    ata[r][c] += basis[r] * basis[c]

        coeffs = _solve_linear_system(ata, aty)

        # Evaluate polynomial at x=0 (center)
        smoothed_value = coeffs[0]
        result.append(smoothed_value)

    return result


def _kmeans(features: List[List[float]], n_clusters: int, max_iter: int = 50) -> Tuple[List[int], List[List[float]]]:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if len(features) < n_clusters:
        raise ValueError("Not enough feature samples for clustering")

    # Initialize with random unique samples
    centers = [features[i][:] for i in random.sample(range(len(features)), n_clusters)]
    labels = [0] * len(features)

    for _ in range(max_iter):
        moved = False
        # Assignment step
        for idx, feat in enumerate(features):
            distances = [
                math.sqrt(sum((feat[d] - center[d]) ** 2 for d in range(len(feat))))
                for center in centers
            ]
            best_cluster = distances.index(min(distances))
            if labels[idx] != best_cluster:
                moved = True
                labels[idx] = best_cluster

        # Update step
        new_centers: List[List[float]] = [[0.0 for _ in features[0]] for _ in range(n_clusters)]
        counts = [0] * n_clusters
        for label, feat in zip(labels, features):
            counts[label] += 1
            for d in range(len(feat)):
                new_centers[label][d] += feat[d]

        for k in range(n_clusters):
            if counts[k] == 0:
                new_centers[k] = centers[k][:]
            else:
                new_centers[k] = [value / counts[k] for value in new_centers[k]]

        centers = new_centers

        if not moved:
            break

    return labels, centers


# =============================================================================
# Core motion analysis
# =============================================================================

class MotionAnalyzerScratch:
    def __init__(self, fps: float):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 0.0

    def smooth_path(
        self,
        path: List[Vector2],
        method: str = "poly",
        window: int = 7,
        poly_order: int = 2,
    ) -> List[Vector2]:
        if len(path) < 2:
            return list(path)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        if method == "moving_average":
            smooth_x = _moving_average(xs, window)
            smooth_y = _moving_average(ys, window)
        elif method == "poly":
            smooth_x = _local_polynomial(xs, window, poly_order)
            smooth_y = _local_polynomial(ys, window, poly_order)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return list(zip(smooth_x, smooth_y))

    def compute_derivatives(self, smooth_path: List[Vector2]) -> Dict[str, List[Vector2]]:
        if len(smooth_path) < 2 or self.dt == 0.0:
            empty = []
            return {"velocity": empty, "acceleration": empty, "jerk": empty, "jounce": empty}

        def diff_series(series: List[Vector2]) -> List[Vector2]:
            return [_vector_scale(_vector_sub(series[i + 1], series[i]), 1.0 / self.dt) for i in range(len(series) - 1)]

        velocity = diff_series(smooth_path)
        acceleration = diff_series(velocity) if len(velocity) > 1 else []
        jerk = diff_series(acceleration) if len(acceleration) > 1 else []
        jounce = diff_series(jerk) if len(jerk) > 1 else []

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "jounce": jounce,
        }

    def compute_scalar_derivatives(self, derivatives: Dict[str, List[Vector2]]) -> Dict[str, List[float]]:
        return {
            "speed": [_vector_norm(v) for v in derivatives.get("velocity", [])],
            "accel_mag": [_vector_norm(v) for v in derivatives.get("acceleration", [])],
            "jerk_mag": [_vector_norm(v) for v in derivatives.get("jerk", [])],
            "jounce_mag": [_vector_norm(v) for v in derivatives.get("jounce", [])],
        }


class SpriteTrackerScratch(MotionAnalyzerScratch):
    def __init__(self, fps: float, templates, pixels_per_meter: float = 32.0):
        super().__init__(fps)
        self.templates = []
        if isinstance(templates, list):
            for t in templates:
                gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if len(t.shape) == 3 else t
                self.templates.append(gray)
        else:
            gray = cv2.cvtColor(templates, cv2.COLOR_BGR2GRAY) if len(templates.shape) == 3 else templates
            self.templates.append(gray)
        self.pixels_per_meter = pixels_per_meter
        self.path: List[Vector2] = []

    def detect_sprite(self, frame, threshold: float = 0.4) -> Optional[Vector2]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        best_score = -1.0
        best_loc: Optional[Tuple[int, int]] = None
        best_shape: Optional[Tuple[int, int]] = None

        for template in self.templates:
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_shape = template.shape

        if best_loc is None or best_shape is None or best_score < threshold:
            return None

        h, w = best_shape
        return (best_loc[0] + w // 2, best_loc[1] + h // 2)

    def track_video(self, video_path: str, threshold: float = 0.4) -> List[Vector2]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.path = []
        fallback_count = 0
        max_fallback = 8

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pos = self.detect_sprite(frame, threshold)
            if pos is not None:
                self.path.append((float(pos[0]), float(pos[1])))
                fallback_count = 0
            elif self.path and fallback_count < max_fallback:
                self.path.append(self.path[-1])
                fallback_count += 1

        cap.release()

        if not self.path:
            raise ValueError("No sprite detections found.")
        return self.path

    def cluster_motion_states(
        self,
        window_size: int = 20,
        n_clusters: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[int], List[List[float]], List[List[float]]]:
        if not self.path:
            raise ValueError("No tracking data available.")

        smooth_path = self.smooth_path(self.path)
        derivatives = self.compute_derivatives(smooth_path)
        scalars = self.compute_scalar_derivatives(derivatives)

        speeds = scalars["speed"]
        accels = scalars["accel_mag"]
        jerks = scalars["jerk_mag"]

        if len(speeds) < window_size:
            raise ValueError("Not enough samples for clustering window.")

        step = max(1, window_size // 2)
        features: List[List[float]] = []

        for start in range(0, len(speeds) - window_size + 1, step):
            speed_window = speeds[start : start + window_size]
            accel_window = accels[start : min(start + window_size - 1, len(accels))]
            jerk_window = jerks[start : min(start + window_size - 2, len(jerks))]

            avg_speed = _mean(speed_window)
            max_accel = max(accel_window) if accel_window else 0.0
            avg_jerk = _mean(jerk_window) if jerk_window else 0.0

            features.append([avg_speed, max_accel, avg_jerk])

        if weights:
            for vec in features:
                vec[0] *= weights.get("speed", 1.0)
                vec[1] *= weights.get("accel", 1.0)
                vec[2] *= weights.get("jerk", 1.0)

        # Normalize manually
        transposed = list(zip(*features))
        means = [_mean(component) for component in transposed]
        stds = [_std(component) for component in transposed]

        normalized: List[List[float]] = []
        for vec in features:
            normalized.append([(vec[i] - means[i]) / stds[i] for i in range(3)])

        labels, centers = _kmeans(normalized, n_clusters)
        return labels, features, centers

    def convert_to_physical_units(self, derivatives: Dict[str, List[Vector2]]) -> Dict[str, List[Vector2]]:
        factor = 1.0 / self.pixels_per_meter if self.pixels_per_meter else 1.0
        return {key: [_vector_scale(vec, factor) for vec in value] for key, value in derivatives.items()}


class KalmanFilterScratch:
    def __init__(self, dt: float, process_noise: float = 1.0, measurement_noise: float = 1.0):
        self.dt = dt
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.P = [
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
        ]
        self.Q = [[process_noise if i == j else 0.0 for j in range(4)] for i in range(4)]
        self.R = [[measurement_noise if i == j else 0.0 for j in range(2)] for i in range(2)]
        self.F = [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        self.H = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]

    def predict(self):
        new_state = [0.0, 0.0, 0.0, 0.0]
        for row in range(4):
            new_state[row] = sum(self.F[row][col] * self.state[col] for col in range(4))
        self.state = new_state

        new_P = [[0.0 for _ in range(4)] for _ in range(4)]
        for r in range(4):
            for c in range(4):
                new_P[r][c] = sum(self.F[r][k] * self.P[k][c] for k in range(4))
        updated_P = [[0.0 for _ in range(4)] for _ in range(4)]
        for r in range(4):
            for c in range(4):
                updated_P[r][c] = sum(new_P[r][k] * self.F[c][k] for k in range(4)) + self.Q[r][c]
        self.P = updated_P

    def update(self, measurement: Vector2):
        z = list(measurement)
        y = [z[i] - sum(self.H[i][j] * self.state[j] for j in range(4)) for i in range(2)]

        S = [[0.0, 0.0], [0.0, 0.0]]
        for r in range(2):
            for c in range(2):
                S[r][c] = sum(self.H[r][k] * self.P[k][c] for k in range(4)) + self.R[r][c]

        # Compute Kalman gain via manual 2x2 inversion
        det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        inv_S = [
            [S[1][1] / det, -S[0][1] / det],
            [-S[1][0] / det, S[0][0] / det],
        ]
        PHt = [[sum(self.P[row][k] * self.H[col][k] for k in range(4)) for col in range(2)] for row in range(4)]
        K = [[sum(PHt[row][i] * inv_S[i][col] for i in range(2)) for col in range(2)] for row in range(4)]

        for i in range(4):
            self.state[i] += sum(K[i][j] * y[j] for j in range(2))

        I = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        KH = [[sum(K[i][k] * self.H[k][j] for k in range(2)) for j in range(4)] for i in range(4)]
        IKH = [[I[i][j] - KH[i][j] for j in range(4)] for i in range(4)]

        new_P = [[0.0 for _ in range(4)] for _ in range(4)]
        for r in range(4):
            for c in range(4):
                new_P[r][c] = sum(IKH[r][k] * self.P[k][c] for k in range(4))
        self.P = new_P

    def get_position(self) -> Vector2:
        return (self.state[0], self.state[1])


class FlockTrackerScratch(MotionAnalyzerScratch):
    def __init__(self, fps: float):
        super().__init__(fps)
        self.paths: List[List[Vector2]] = []

    def detect_objects(self, frame, threshold: int = 50, min_area: int = 5, max_area: int = 500) -> List[Vector2]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids: List[Vector2] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((cx, cy))
        return centroids

    def associate_detections(
        self,
        prev_positions: List[Vector2],
        detections: List[Vector2],
        max_distance: float = 60.0,
    ) -> List[Optional[int]]:
        assignments: List[Optional[int]] = [None] * len(prev_positions)
        used: set[int] = set()

        for idx, prev in enumerate(prev_positions):
            best_j = None
            best_d = float("inf")
            for j, det in enumerate(detections):
                if j in used:
                    continue
                dist = _vector_norm(_vector_sub(prev, det))
                if dist < best_d:
                    best_d = dist
                    best_j = j
            if best_j is not None and best_d <= max_distance:
                assignments[idx] = best_j
                used.add(best_j)
        return assignments

    def track_video(self, video_path: str, threshold: int = 50, use_kalman: bool = True) -> List[List[Vector2]]:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            return []

        detections = self.detect_objects(frame, threshold)
        self.paths = [[det] for det in detections]
        kalman_filters = [KalmanFilterScratch(self.dt) for _ in detections] if use_kalman else []
        for kf, det in zip(kalman_filters, detections):
            kf.state[0], kf.state[1] = det

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detect_objects(frame, threshold)

            if use_kalman:
                for kf in kalman_filters:
                    kf.predict()
                predictions = [kf.get_position() for kf in kalman_filters]
            else:
                predictions = [path[-1] for path in self.paths]

            assignments = self.associate_detections(predictions, detections)

            for idx, assignment in enumerate(assignments):
                if assignment is not None:
                    det = detections[assignment]
                    if use_kalman:
                        kalman_filters[idx].update(det)
                        self.paths[idx].append(kalman_filters[idx].get_position())
                    else:
                        self.paths[idx].append(det)
                else:
                    self.paths[idx].append(predictions[idx])

        cap.release()
        return self.paths

    def cluster_by_behavior(
        self,
        n_clusters: int = 2,
        weights: Optional[Dict[str, float]] = None,
        min_path_length: int = 10,
    ) -> Tuple[List[int], List[List[float]], List[List[float]]]:
        if not self.paths:
            raise ValueError("No tracking paths available.")

        features: List[List[float]] = []
        for path in self.paths:
            if len(path) < min_path_length:
                features.append([0.0, 0.0, 0.0])
                continue

            smooth = self.smooth_path(path)
            derivatives = self.compute_derivatives(smooth)
            scalars = self.compute_scalar_derivatives(derivatives)

            avg_speed = _mean(scalars["speed"]) if scalars["speed"] else 0.0
            avg_accel = _mean(scalars["accel_mag"]) if scalars["accel_mag"] else 0.0
            avg_jerk = _mean(scalars["jerk_mag"]) if scalars["jerk_mag"] else 0.0

            features.append([avg_speed, avg_accel, avg_jerk])

        if weights:
            for vec in features:
                vec[0] *= weights.get("speed", 1.0)
                vec[1] *= weights.get("accel", 1.0)
                vec[2] *= weights.get("jerk", 1.0)

        transposed = list(zip(*features))
        means = [_mean(c) for c in transposed]
        stds = [_std(c) for c in transposed]
        normalized = [[(vec[i] - means[i]) / stds[i] for i in range(3)] for vec in features]

        labels, centers = _kmeans(normalized, min(n_clusters, len(features)))
        return labels, features, centers


def visualize_sprite_tracking(tracker: SpriteTrackerScratch):
    """
    Placeholder visualization hook — relies on matplotlib in the caller
    to keep this module dependency-light. Returns raw data needed for plotting.
    """
    path = tracker.path
    smooth = tracker.smooth_path(path)
    derivatives = tracker.compute_derivatives(smooth)
    scalars = tracker.compute_scalar_derivatives(derivatives)
    return {
        "path": path,
        "smooth_path": smooth,
        "scalars": scalars,
    }


def visualize_flock_tracking(tracker: FlockTrackerScratch, cluster_labels: Optional[List[int]] = None):
    return {
        "paths": tracker.paths,
        "cluster_labels": cluster_labels or [],
    }


if __name__ == "__main__":
    print("Scratch-built Motion Tracker")
    print("- Vector math and smoothing implemented manually")
    print("- No NumPy/SciPy/SKLearn dependencies for analytics")

