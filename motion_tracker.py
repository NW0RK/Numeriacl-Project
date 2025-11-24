import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


class MotionAnalyzer:
    """Base class for motion analysis with derivatives and clustering."""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.dt = 1.0 / fps
        
    def smooth_path(self, path: np.ndarray, method: str = 'savgol', 
                    window: int = 11, poly_order: int = 3) -> np.ndarray:
        """
        Smooth a noisy path to reduce tracking jitter and improve derivative accuracy.
        
        Savitzky-Golay: Fits polynomial to local window, preserves features better
        Gaussian: Simple blur, faster but may oversmooth sharp changes
        
        Args:
            path: Array of shape (n_frames, 2) with (x, y) positions
            method: 'savgol' (recommended) or 'gaussian'
            window: Window size for smoothing (must be odd for savgol, larger = more smoothing)
            poly_order: Polynomial order for Savitzky-Golay (typically 2-4)
            
        Returns:
            Smoothed path of same shape (n_frames, 2)
        """
        if len(path) == 0:
            return path
        
        # If path is shorter than window, return as-is (can't smooth)
        if len(path) < window:
            return path
        
        # Ensure window is odd for Savitzky-Golay (required by scipy)
        if method == 'savgol' and window % 2 == 0:
            window = window - 1
            if window < 3:
                window = 3
        
        # Ensure poly_order doesn't exceed window size
        if method == 'savgol' and poly_order >= window:
            poly_order = max(1, window - 1)
            
        if method == 'savgol':
            # Savitzky-Golay: fits polynomial to local window, preserves derivatives
            # Better for maintaining motion characteristics while reducing noise
            smooth_x = savgol_filter(path[:, 0], window, poly_order)
            smooth_y = savgol_filter(path[:, 1], window, poly_order)
        elif method == 'gaussian':
            # Gaussian: simple convolution with Gaussian kernel
            # Faster but may blur sharp motion changes
            sigma = window / 6.0  # Convert window size to sigma (3-sigma rule)
            smooth_x = gaussian_filter1d(path[:, 0], sigma)
            smooth_y = gaussian_filter1d(path[:, 1], sigma)
        else:
            raise ValueError(f"Unknown smoothing method: {method}. Use 'savgol' or 'gaussian'")
            
        return np.column_stack([smooth_x, smooth_y])
    
    def compute_derivatives(self, smooth_path: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute 1st through 4th order derivatives of a smoothed path.
        
        Derivatives are computed using finite differences:
        - Velocity: rate of position change (dx/dt, dy/dt)
        - Acceleration: rate of velocity change (d²x/dt², d²y/dt²)
        - Jerk: rate of acceleration change (d³x/dt³, d³y/dt³)
        - Jounce: rate of jerk change (d⁴x/dt⁴, d⁴y/dt⁴)
        
        Note: Each derivative has one fewer sample than the previous due to diff().
        For n frames, you get: n-1 velocities, n-2 accelerations, n-3 jerks, n-4 jounces.
        
        Args:
            smooth_path: Array of shape (n_frames, 2) with smoothed (x, y) positions
            
        Returns:
            Dictionary with 'velocity', 'acceleration', 'jerk', 'jounce'
            Each is an array of shape (n_frames-k, 2) where k is the derivative order
        """
        # Handle edge case: need at least 2 points for velocity
        if len(smooth_path) < 2:
            return {
                'velocity': np.empty((0, 2)),
                'acceleration': np.empty((0, 2)),
                'jerk': np.empty((0, 2)),
                'jounce': np.empty((0, 2))
            }

        # 1st derivative: velocity (pixels/second)
        # np.diff computes differences between consecutive points
        velocity = np.diff(smooth_path, axis=0) / self.dt
        
        # 2nd derivative: acceleration (pixels/second²)
        # Change in velocity over time
        acceleration = np.diff(velocity, axis=0) / self.dt
        
        # 3rd derivative: jerk (pixels/second³)
        # Change in acceleration - useful for detecting sudden motion changes (jumps)
        jerk = np.diff(acceleration, axis=0) / self.dt
        
        # 4th derivative: jounce/snap (pixels/second⁴)
        # Rarely used but included for completeness
        jounce = np.diff(jerk, axis=0) / self.dt
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'jounce': jounce
        }
    
    def compute_scalar_derivatives(self, derivatives: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert vector derivatives to scalar magnitudes (Euclidean norm).
        
        Vector derivatives have both magnitude and direction. For many analyses,
        we only care about magnitude (e.g., "how fast?" not "which direction?").
        
        Examples:
        - speed = ||velocity|| = sqrt(vx² + vy²)
        - accel_mag = ||acceleration|| = sqrt(ax² + ay²)
        
        Args:
            derivatives: Dictionary of vector derivatives (each is shape (n, 2))
            
        Returns:
            Dictionary with scalar magnitudes:
            - speed: magnitude of velocity (pixels/s)
            - accel_mag: magnitude of acceleration (pixels/s²)
            - jerk_mag: magnitude of jerk (pixels/s³)
            - jounce_mag: magnitude of jounce (pixels/s⁴)
        """
        if len(derivatives['velocity']) == 0:
            return {
                'speed': np.array([]),
                'accel_mag': np.array([]),
                'jerk_mag': np.array([]),
                'jounce_mag': np.array([])
            }

        # Compute Euclidean norm (magnitude) for each derivative
        # axis=1 means compute norm across x and y components for each time step
        return {
            'speed': np.linalg.norm(derivatives['velocity'], axis=1),
            'accel_mag': np.linalg.norm(derivatives['acceleration'], axis=1),
            'jerk_mag': np.linalg.norm(derivatives['jerk'], axis=1),
            'jounce_mag': np.linalg.norm(derivatives['jounce'], axis=1)
        }


class SpriteTracker(MotionAnalyzer):
    """Tracker for single 2D sprite (e.g., Super Mario Bros.)"""
    
    def __init__(self, fps: float, template, pixels_per_meter: float = 32.0):
        super().__init__(fps)
        
        # Handle single template or list of templates
        if isinstance(template, list):
            self.templates = []
            for t in template:
                gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if len(t.shape) == 3 else t
                self.templates.append(gray)
        else:
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            self.templates = [gray]
            
        self.pixels_per_meter = pixels_per_meter
        self.path = []
        self.matched_template_ids = []  # Track which template matched each frame
        
    def detect_sprite(self, frame: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Detect sprite position using template matching (normalized cross-correlation).
        
        Supports multiple templates (e.g., different animation frames) and returns
        the best match across all templates. Uses TM_CCOEFF_NORMED which is robust
        to lighting changes.
        
        Args:
            frame: Video frame (BGR or grayscale)
            threshold: Minimum correlation score (0-1), higher = more strict matching
                       Typical values: 0.6-0.8 for good matches, 0.4-0.6 for permissive
            
        Returns:
            (x, y) position of sprite center in pixels, or None if no match above threshold
        """
        # Convert to grayscale if needed (template matching works on grayscale)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        best_match_score = -1.0
        best_match_loc = None
        best_template_shape = None
        
        # Loop through each template to find the best match
        # Useful when sprite has multiple appearances (e.g., facing left/right, jumping)
        for template in self.templates:
            # Skip if template is larger than frame (can't match)
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue

            # Template matching: slides template over image, computes correlation at each position
            # TM_CCOEFF_NORMED: normalized cross-correlation coefficient (0-1, higher = better match)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Keep track of best match across all templates
            if max_val > best_match_score:
                best_match_score = max_val
                best_match_loc = max_loc  # Top-left corner of match
                best_template_shape = template.shape
        
        # Return center position if match is above threshold
        if best_match_score >= threshold:
            h, w = best_template_shape
            center_x = best_match_loc[0] + w // 2
            center_y = best_match_loc[1] + h // 2
            return (center_x, center_y)
        
        return None
    
    def track_video(self, video_path: str, threshold: float = 0.8) -> np.ndarray:
        """
        Track sprite through entire video using template matching.
        
        When detection fails, uses last known position to maintain continuity.
        For better handling of occlusions, consider using Kalman filtering.
        
        Args:
            video_path: Path to video file
            threshold: Detection threshold (0-1), lower = more permissive
            
        Returns:
            Array of shape (n_frames, 2) with (x, y) positions in pixels
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.path = []
        consecutive_failures = 0
        max_failures = 10  # Maximum frames to keep using last position
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            position = self.detect_sprite(frame, threshold)
            if position is not None:
                self.path.append(position)
                consecutive_failures = 0
            elif len(self.path) > 0 and consecutive_failures < max_failures:
                # If detection fails, use last known position (simple approach)
                # For better results, consider linear interpolation or Kalman filter
                self.path.append(self.path[-1])
                consecutive_failures += 1
            elif len(self.path) == 0:
                # First frame failed - skip it
                continue
            # If too many failures, stop appending (sprite likely disappeared)
                
        cap.release()
        
        if len(self.path) == 0:
            raise ValueError("No detections found in video. Try lowering threshold or checking templates.")
        
        return np.array(self.path)
    
    def cluster_motion_states(self, window_size: int = 20, n_clusters: int = 3,
                             weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Cluster motion into behavioral states (standing, running, jumping).
        
        Uses sliding windows to extract motion features (speed, acceleration, jerk)
        and applies K-Means clustering to identify distinct motion patterns.
        
        Args:
            window_size: Number of frames per segment (must be >= 3 for jerk calculation)
            n_clusters: Number of motion states to find (e.g., 3 for standing/running/jumping)
            weights: Dictionary with weights for 'speed', 'accel', 'jerk' to emphasize
                    certain features (e.g., {'jerk': 10.0} to emphasize jumping)
            
        Returns:
            Tuple of (labels, features, centers):
            - labels: Array of cluster labels for each time window
            - features: Array of feature vectors (n_windows, 3) with [speed, accel, jerk]
            - centers: Cluster centers from K-Means
        """
        if len(self.path) == 0:
            raise ValueError("No path data. Run track_video first.")
        
        # Validate window size (need at least 3 frames for jerk calculation)
        if window_size < 3:
            raise ValueError(f"window_size must be >= 3, got {window_size}")
            
        # Smooth path to reduce noise before computing derivatives
        smooth_path = self.smooth_path(np.array(self.path))
        
        # Compute derivatives: velocity (1st), acceleration (2nd), jerk (3rd)
        # Note: Each derivative has one fewer sample than the previous
        derivatives = self.compute_derivatives(smooth_path)
        scalars = self.compute_scalar_derivatives(derivatives)
        
        # Check if we have enough data for at least one window
        if len(scalars['speed']) < window_size:
            raise ValueError(f"Not enough data: need at least {window_size} speed samples, "
                           f"got {len(scalars['speed'])}")
        
        # Create feature vectors for each sliding window
        # Windows overlap by 50% (step = window_size // 2) for smoother transitions
        features = []
        step_size = max(1, window_size // 2)  # Ensure step is at least 1
        
        for i in range(0, len(scalars['speed']) - window_size + 1, step_size):
            # Define window slices (accel and jerk need smaller windows due to derivative loss)
            window = slice(i, i + window_size)
            window_accel = slice(i, min(i + window_size - 1, len(scalars['accel_mag'])))
            window_jerk = slice(i, min(i + window_size - 2, len(scalars['jerk_mag'])))
            
            # Extract features: average speed, max acceleration, average jerk
            # Max acceleration captures sudden changes (jumps, stops)
            # Average jerk captures smoothness of motion changes
            avg_speed = np.mean(scalars['speed'][window])
            max_accel = np.max(scalars['accel_mag'][window_accel]) if len(scalars['accel_mag'][window_accel]) > 0 else 0.0
            avg_jerk = np.mean(scalars['jerk_mag'][window_jerk]) if len(scalars['jerk_mag'][window_jerk]) > 0 else 0.0
            
            features.append([avg_speed, max_accel, avg_jerk])
        
        if len(features) == 0:
            raise ValueError("No feature windows could be created. Check window_size and data length.")
        
        features = np.array(features)
        
        # Apply feature weights if provided (allows custom distance metrics)
        # Example: weights={'jerk': 10.0} emphasizes jerk for detecting jumps
        if weights:
            w = np.array([
                weights.get('speed', 1.0),
                weights.get('accel', 1.0),
                weights.get('jerk', 1.0)
            ])
            features = features * w
        
        # Normalize features to prevent one feature from dominating
        # (optional but recommended for better clustering)
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        # Avoid division by zero
        feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)
        features_normalized = (features - feature_means) / feature_stds
        
        # Validate: need at least n_clusters samples for clustering
        n_samples = len(features_normalized)
        if n_samples < n_clusters:
            # Automatically reduce n_clusters to match available samples
            import warnings
            warnings.warn(
                f"Only {n_samples} sample(s) available, but {n_clusters} clusters requested. "
                f"Reducing to {n_samples} cluster(s).",
                UserWarning
            )
            # If only 1 sample, assign it to cluster 0
            if n_samples == 1:
                labels = np.array([0])
                centers = features_normalized.reshape(1, -1)
            else:
                # Use all samples as clusters (each sample is its own cluster)
                labels = np.arange(n_samples)
                centers = features_normalized
            return labels, features, centers
        
        # K-Means clustering to identify motion states
        # random_state=42 for reproducibility, n_init=10 for better convergence
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        return labels, features, kmeans.cluster_centers_
    
    def convert_to_physical_units(self, derivatives: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert pixel-based derivatives to physical units (meters, m/s, etc.)
        
        Args:
            derivatives: Dictionary with pixel-based derivatives
            
        Returns:
            Dictionary with physical unit derivatives
        """
        physical = {
            'velocity': derivatives['velocity'] / self.pixels_per_meter,
            'acceleration': derivatives['acceleration'] / self.pixels_per_meter,
            'jerk': derivatives['jerk'] / self.pixels_per_meter,
            'jounce': derivatives['jounce'] / self.pixels_per_meter
        }
        
        return physical


class KalmanFilter:
    """
    Simple 2D Kalman Filter for robust tracking with prediction.
    
    Uses constant velocity model: state = [x, y, vx, vy]
    Predicts position between detections and smooths noisy measurements.
    Useful for handling temporary occlusions or detection failures.
    """
    
    def __init__(self, dt: float, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        Initialize Kalman filter with constant velocity model.
        
        Args:
            dt: Time step (1/fps)
            process_noise: Uncertainty in motion model (higher = more prediction uncertainty)
            measurement_noise: Uncertainty in detections (higher = trust predictions more)
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        
        # State transition matrix F: predicts next state from current state
        # x' = x + vx*dt, y' = y + vy*dt, vx' = vx, vy' = vy (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],  # x_new = x + vx*dt
            [0, 1, 0, dt],  # y_new = y + vy*dt
            [0, 0, 1, 0],   # vx_new = vx (constant)
            [0, 0, 0, 1]    # vy_new = vy (constant)
        ])
        
        # Measurement matrix H: maps state to measurements
        # We only observe position [x, y], not velocity
        self.H = np.array([
            [1, 0, 0, 0],  # x_measured = x
            [0, 1, 0, 0]   # y_measured = y
        ])
        
        # Covariance matrices
        self.P = np.eye(4) * 100  # State covariance (uncertainty in state estimate)
        self.Q = np.eye(4) * process_noise  # Process noise (uncertainty in motion model)
        self.R = np.eye(2) * measurement_noise  # Measurement noise (uncertainty in detections)
        
    def predict(self):
        """
        Predict next state using motion model (constant velocity).
        
        Called before each measurement to propagate state forward in time.
        Increases uncertainty (P) because prediction is less certain than measurement.
        """
        # Predict state: x' = F @ x
        self.state = self.F @ self.state
        
        # Predict covariance: P' = F @ P @ F.T + Q
        # Uncertainty grows due to process noise Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """
        Update state with measurement using Kalman filter update step.
        
        Uses numerically stable matrix solve instead of explicit inverse.
        
        Args:
            measurement: Observed position [x, y]
        """
        # Innovation: difference between measurement and predicted measurement
        y = measurement - self.H @ self.state
        
        # Innovation covariance: uncertainty in the measurement prediction
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: how much to trust the measurement vs prediction
        # Use solve() instead of inv() for numerical stability and efficiency
        # Solve S @ K.T = (P @ H.T).T, then transpose to get K
        K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        
        # Update state estimate: blend prediction and measurement
        self.state = self.state + K @ y
        
        # Update covariance: reduce uncertainty after measurement
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2]


class FlockTracker(MotionAnalyzer):
    """Tracker for multiple objects (e.g., starling flock)"""
    
    def __init__(self, fps: float, pixels_per_meter: Optional[float] = None):
        super().__init__(fps)
        self.paths = []  # List of paths, one per bird
        self.n_objects = 0
        self.pixels_per_meter = pixels_per_meter

    def set_pixels_per_meter(self, pixels_per_meter: float) -> None:
        """Assign a global scene scale for pixel → meter conversion."""
        if pixels_per_meter is None or pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be a positive number")
        self.pixels_per_meter = float(pixels_per_meter)
        
    def detect_objects(self, frame: np.ndarray, threshold: int = 50,
                       min_area: int = 5, max_area: int = 500) -> List[Tuple[int, int]]:
        """
        Detect objects using thresholding and contour detection.
        
        Args:
            frame: Video frame
            threshold: Binary threshold value
            min_area: Minimum contour area
            max_area: Maximum contour area
            
        Returns:
            List of (x, y) centroids
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Binary threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract centroids
        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    
        return centroids
    
    def estimate_pixels_per_meter(self, frame: np.ndarray, threshold: int = 50,
                                  min_area: int = 5, max_area: int = 500,
                                  reference_size_m: float = 0.35) -> Optional[float]:
        """
        Estimate pixels-per-meter automatically using the average blob diameter.
        
        Args:
            frame: Video frame used for calibration
            threshold: Binary threshold value
            min_area: Minimum contour area
            max_area: Maximum contour area
            reference_size_m: Real-world size (meters) of a typical object (e.g., wingspan)
        
        Returns:
            Estimated pixels-per-meter value, or None if estimation failed.
        """
        if reference_size_m <= 0:
            return None
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        diameters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Equivalent diameter from contour area
                diameter = 2.0 * np.sqrt(area / np.pi)
                diameters.append(diameter)
        
        if len(diameters) == 0:
            return None
        
        median_diameter = float(np.median(diameters))
        return median_diameter / reference_size_m
    
    def associate_detections(self, prev_positions: List[np.ndarray], 
                            curr_centroids: List[Tuple[int, int]],
                            max_distance: float = 50.0) -> List[Optional[int]]:
        """
        Associate current detections with previous tracks using nearest neighbor matching.
        
        This solves the data association problem: which detection corresponds to which track?
        Uses greedy nearest neighbor: each track is matched to its closest unused detection.
        More sophisticated methods (Hungarian algorithm, global nearest neighbor) could be used
        for better accuracy with many objects.
        
        Args:
            prev_positions: List of previous positions (one per track)
            curr_centroids: List of current detections (x, y) tuples
            max_distance: Maximum association distance in pixels (gating threshold)
            
        Returns:
            List mapping each previous track to current centroid index (or None if lost)
        """
        if len(prev_positions) == 0 or len(curr_centroids) == 0:
            return [None] * len(prev_positions)
            
        # Compute distance matrix: distances[i, j] = distance from track i to detection j
        distances = np.zeros((len(prev_positions), len(curr_centroids)))
        for i, prev_pos in enumerate(prev_positions):
            for j, curr_pos in enumerate(curr_centroids):
                distances[i, j] = np.linalg.norm(prev_pos - np.array(curr_pos))
        
        # Greedy assignment: each track gets its nearest unused detection
        # Note: This is suboptimal but fast. For better results with many objects,
        # use Hungarian algorithm (scipy.optimize.linear_sum_assignment)
        assignments = [None] * len(prev_positions)
        used_centroids = set()  # Track which detections are already assigned
        
        for i in range(len(prev_positions)):
            # Find nearest unused centroid for this track
            valid_distances = distances[i].copy()
            for j in used_centroids:
                valid_distances[j] = float('inf')  # Mark used detections as unavailable
                
            # Only assign if within max_distance (gating)
            if valid_distances.min() < max_distance:
                j = valid_distances.argmin()
                assignments[i] = j
                used_centroids.add(j)
            # If no detection within max_distance, assignment remains None (track lost)
                
        return assignments
    
    def track_video(self, video_path: str, threshold: int = 50,
                   use_kalman: bool = True) -> List[np.ndarray]:
        """
        Track multiple objects through video.
        
        Args:
            video_path: Path to video file
            threshold: Detection threshold
            use_kalman: Whether to use Kalman filtering
            
        Returns:
            List of paths, one per tracked object
        """
        cap = cv2.VideoCapture(video_path)
        
        # Initialize on first frame
        ret, frame = cap.read()
        if not ret:
            return []
            
        first_centroids = self.detect_objects(frame, threshold)
        self.n_objects = len(first_centroids)
        
        # Initialize paths and Kalman filters
        self.paths = [[np.array(c)] for c in first_centroids]
        
        if use_kalman:
            kalman_filters = [KalmanFilter(self.dt) for _ in range(self.n_objects)]
            for kf, centroid in zip(kalman_filters, first_centroids):
                kf.state[:2] = centroid
        
        # Track through video
        frame_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            curr_centroids = self.detect_objects(frame, threshold)
            
            # Get previous positions (from Kalman or raw path)
            if use_kalman:
                # Predict
                for kf in kalman_filters:
                    kf.predict()
                prev_positions = [kf.get_position() for kf in kalman_filters]
            else:
                prev_positions = [path[-1] for path in self.paths]
            
            # Associate
            assignments = self.associate_detections(prev_positions, curr_centroids)
            
            # Update paths
            for i, assignment in enumerate(assignments):
                if assignment is not None:
                    new_pos = np.array(curr_centroids[assignment])
                    if use_kalman:
                        kalman_filters[i].update(new_pos)
                        self.paths[i].append(kalman_filters[i].get_position())
                    else:
                        self.paths[i].append(new_pos)
                else:
                    # Lost track, use prediction or repeat last position
                    if use_kalman:
                        self.paths[i].append(kalman_filters[i].get_position())
                    else:
                        self.paths[i].append(self.paths[i][-1])
            
            frame_count += 1
                    
        cap.release()
        
        # Convert to numpy arrays
        self.paths = [np.array(path) for path in self.paths]
        return self.paths
    
    def aggregate_scalar_metrics(self, min_path_length: int = 5) -> Dict[str, np.ndarray]:
        """
        Aggregate scalar derivatives (speed, accel, jerk) across all valid tracks.
        
        Args:
            min_path_length: Minimum number of samples required to compute derivatives.
        
        Returns:
            Dictionary mapping metric name to concatenated samples (pixels-based).
        """
        metrics = {
            'speed': [],
            'accel_mag': [],
            'jerk_mag': []
        }
        
        for path in self.paths:
            if len(path) < min_path_length:
                continue
            
            smooth_path = self.smooth_path(path)
            derivatives = self.compute_derivatives(smooth_path)
            scalars = self.compute_scalar_derivatives(derivatives)
            
            for key in metrics.keys():
                if len(scalars[key]) > 0:
                    metrics[key].append(scalars[key])
        
        return {
            key: np.concatenate(values) if len(values) > 0 else np.array([])
            for key, values in metrics.items()
        }
    
    def cluster_by_behavior(self, n_clusters: int = 2, 
                           weights: Optional[Dict[str, float]] = None,
                           min_path_length: int = 10) -> np.ndarray:
        """
        Cluster objects by their motion behavior (e.g., leaders vs followers).
        
        Computes average motion characteristics for each object's entire trajectory
        and clusters them based on these features.
        
        Args:
            n_clusters: Number of clusters (e.g., 2 for leaders vs followers)
            weights: Weights for 'speed', 'accel', 'jerk' to emphasize certain behaviors
            min_path_length: Minimum path length to include in clustering (shorter = noise)
            
        Returns:
            Tuple of (labels, features, centers):
            - labels: Array of cluster labels (one per object)
            - features: Array of feature vectors (n_objects, 3)
            - centers: Cluster centers from K-Means
        """
        if len(self.paths) == 0:
            raise ValueError("No path data. Run track_video first.")
            
        # Compute feature vector for each object based on its entire trajectory
        features = []
        valid_indices = []  # Track which paths are valid
        
        for idx, path in enumerate(self.paths):
            if len(path) < min_path_length:
                # Skip very short tracks (likely noise or tracking failures)
                # Use zero features so they can still be assigned to a cluster
                features.append([0.0, 0.0, 0.0])
                valid_indices.append(False)
                continue
            
            valid_indices.append(True)
            
            # Smooth path to reduce noise before computing derivatives
            smooth_path = self.smooth_path(path)
            derivatives = self.compute_derivatives(smooth_path)
            scalars = self.compute_scalar_derivatives(derivatives)
            
            # Aggregate motion characteristics over entire trajectory
            # Average values capture overall behavior pattern
            avg_speed = np.mean(scalars['speed']) if len(scalars['speed']) > 0 else 0.0
            avg_accel = np.mean(scalars['accel_mag']) if len(scalars['accel_mag']) > 0 else 0.0
            avg_jerk = np.mean(scalars['jerk_mag']) if len(scalars['jerk_mag']) > 0 else 0.0
            
            features.append([avg_speed, avg_accel, avg_jerk])
        
        if len(features) == 0:
            raise ValueError("No valid paths for clustering. Check min_path_length.")
        
        features = np.array(features)
        
        # Apply feature weights if provided
        if weights:
            w = np.array([
                weights.get('speed', 1.0),
                weights.get('accel', 1.0),
                weights.get('jerk', 1.0)
            ])
            features = features * w
        
        # Normalize features for better clustering
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)
        features_normalized = (features - feature_means) / feature_stds
        
        # Validate: need at least n_clusters samples for clustering
        n_samples = len(features_normalized)
        if n_samples < n_clusters:
            # Automatically reduce n_clusters to match available samples
            import warnings
            warnings.warn(
                f"Only {n_samples} sample(s) available, but {n_clusters} clusters requested. "
                f"Reducing to {n_samples} cluster(s).",
                UserWarning
            )
            # If only 1 sample, assign it to cluster 0
            if n_samples == 1:
                labels = np.array([0])
                # Create a dummy center (just the single feature)
                centers = features_normalized.reshape(1, -1)
            else:
                # Use all samples as clusters (each sample is its own cluster)
                labels = np.arange(n_samples)
                centers = features_normalized
            return labels, features, centers
        
        # K-Means clustering to identify behavioral groups
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        return labels, features, kmeans.cluster_centers_


# Example usage and visualization functions
def visualize_sprite_tracking(tracker: SpriteTracker):
    """Visualize sprite path and motion analysis."""
    path = np.array(tracker.path)
    smooth_path = tracker.smooth_path(path)
    
    # Compute derivatives
    derivatives = tracker.compute_derivatives(smooth_path)
    scalars = tracker.compute_scalar_derivatives(derivatives)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Path
    axes[0, 0].plot(path[:, 0], path[:, 1], 'b.', alpha=0.3, label='Raw')
    axes[0, 0].plot(smooth_path[:, 0], smooth_path[:, 1], 'r-', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    axes[0, 0].set_title('Sprite Path')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()  # Invert Y for image coordinates
    
    # Plot 2: Speed
    time = np.arange(len(scalars['speed'])) / tracker.fps
    axes[0, 1].plot(time, scalars['speed'])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (pixels/s)')
    axes[0, 1].set_title('Speed Over Time')
    axes[0, 1].grid(True)
    
    # Plot 3: Acceleration magnitude
    time_accel = np.arange(len(scalars['accel_mag'])) / tracker.fps
    axes[1, 0].plot(time_accel, scalars['accel_mag'])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration (pixels/s²)')
    axes[1, 0].set_title('Acceleration Magnitude')
    axes[1, 0].grid(True)
    
    # Plot 4: Jerk magnitude
    time_jerk = np.arange(len(scalars['jerk_mag'])) / tracker.fps
    axes[1, 1].plot(time_jerk, scalars['jerk_mag'])
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Jerk (pixels/s³)')
    axes[1, 1].set_title('Jerk Magnitude')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def visualize_flock_tracking(tracker: FlockTracker, cluster_labels: Optional[np.ndarray] = None):
    """Visualize flock paths and clustering."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All paths
    colors = plt.cm.tab10(np.linspace(0, 1, len(tracker.paths)))
    for i, path in enumerate(tracker.paths):
        if len(path) > 0:
            color = colors[cluster_labels[i]] if cluster_labels is not None else colors[i]
            axes[0].plot(path[:, 0], path[:, 1], alpha=0.6, color=color)
    
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    axes[0].set_title('Flock Trajectories')
    axes[0].invert_yaxis()
    
    # Plot 2: Clustering visualization (if provided)
    if cluster_labels is not None:
        features = []
        for path in tracker.paths:
            if len(path) < 10:
                features.append([0, 0, 0])
                continue
            smooth_path = tracker.smooth_path(path)
            derivatives = tracker.compute_derivatives(smooth_path)
            scalars = tracker.compute_scalar_derivatives(derivatives)
            features.append([
                np.mean(scalars['speed']),
                np.mean(scalars['accel_mag']),
                np.mean(scalars['jerk_mag'])
            ])
        
        features = np.array(features)
        scatter = axes[1].scatter(features[:, 0], features[:, 1], 
                                 c=cluster_labels, cmap='viridis', s=100, alpha=0.6)
        axes[1].set_xlabel('Average Speed')
        axes[1].set_ylabel('Average Acceleration')
        axes[1].set_title('Behavior Clustering')
        plt.colorbar(scatter, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Motion Tracking Implementation")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. SpriteTracker - for single object tracking (e.g., Mario)")
    print("2. FlockTracker - for multiple object tracking (e.g., birds)")
    print("\nBoth support:")
    print("- Path smoothing (Savitzky-Golay, Gaussian)")
    print("- 1st-4th order derivative computation")
    print("- Motion-based clustering with custom norms")
    print("- Kalman filtering for robust tracking")