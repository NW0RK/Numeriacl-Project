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
        Smooth a noisy path using either Savitzky-Golay or Gaussian filtering.
        
        Args:
            path: Array of shape (n_frames, 2) with (x, y) positions
            method: 'savgol' or 'gaussian'
            window: Window size for smoothing
            poly_order: Polynomial order for Savitzky-Golay
            
        Returns:
            Smoothed path of same shape
        """
        if len(path) < window:
            return path
            
        if method == 'savgol':
            # Apply Savitzky-Golay filter to each dimension
            smooth_x = savgol_filter(path[:, 0], window, poly_order)
            smooth_y = savgol_filter(path[:, 1], window, poly_order)
        elif method == 'gaussian':
            # Apply Gaussian filter
            sigma = window / 6.0  # Convert window to sigma
            smooth_x = gaussian_filter1d(path[:, 0], sigma)
            smooth_y = gaussian_filter1d(path[:, 1], sigma)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
            
        return np.column_stack([smooth_x, smooth_y])
    
    def compute_derivatives(self, smooth_path: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute 1st through 4th order derivatives of a smoothed path.
        
        Args:
            smooth_path: Array of shape (n_frames, 2) with smoothed (x, y) positions
            
        Returns:
            Dictionary with 'velocity', 'acceleration', 'jerk', 'jounce'
            Each is an array of shape (n_frames-k, 2) where k is the derivative order
        """
        # 1st derivative: velocity (pixels/second)
        velocity = np.diff(smooth_path, axis=0) / self.dt
        
        # 2nd derivative: acceleration (pixels/second²)
        acceleration = np.diff(velocity, axis=0) / self.dt
        
        # 3rd derivative: jerk (pixels/second³)
        jerk = np.diff(acceleration, axis=0) / self.dt
        
        # 4th derivative: jounce/snap (pixels/second⁴)
        jounce = np.diff(jerk, axis=0) / self.dt
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'jounce': jounce
        }
    
    def compute_scalar_derivatives(self, derivatives: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert vector derivatives to scalar magnitudes.
        
        Args:
            derivatives: Dictionary of vector derivatives
            
        Returns:
            Dictionary with scalar magnitudes (speed, accel_mag, etc.)
        """
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
        Detect sprite position using template matching.
        
        Args:
            frame: Video frame (BGR or grayscale)
            threshold: Minimum correlation score (0-1)
            
        Returns:
            (x, y) position of sprite center, or None if not found
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Template matching
        result = cv2.matchTemplate(gray, self.templates, cv2.TM_CCOEFF_NORMED)
        
        # Find maximum correlation
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            # Get center of template
            h, w = self.template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
        
        return None
    
    def track_video(self, video_path: str, threshold: float = 0.8) -> np.ndarray:
        """
        Track sprite through entire video.
        
        Args:
            video_path: Path to video file
            threshold: Detection threshold
            
        Returns:
            Array of shape (n_frames, 2) with positions
        """
        cap = cv2.VideoCapture(video_path)
        self.path = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            position = self.detect_sprite(frame, threshold)
            if position is not None:
                self.path.append(position)
            elif len(self.path) > 0:
                # If detection fails, use last known position
                self.path.append(self.path[-1])
                
        cap.release()
        return np.array(self.path)
    
    def cluster_motion_states(self, window_size: int = 20, n_clusters: int = 3,
                             weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Cluster motion into behavioral states (standing, running, jumping).
        
        Args:
            window_size: Number of frames per segment
            n_clusters: Number of motion states to find
            weights: Dictionary with weights for 'speed', 'accel', 'jerk'
            
        Returns:
            Array of cluster labels for each time window
        """
        if len(self.path) == 0:
            raise ValueError("No path data. Run track_video first.")
            
        # Smooth path
        smooth_path = self.smooth_path(np.array(self.path))
        
        # Compute derivatives
        derivatives = self.compute_derivatives(smooth_path)
        scalars = self.compute_scalar_derivatives(derivatives)
        
        # Create feature vectors for each window
        features = []
        for i in range(0, len(scalars['speed']) - window_size, window_size // 2):
            window = slice(i, i + window_size)
            
            avg_speed = np.mean(scalars['speed'][window])
            max_accel = np.max(scalars['accel_mag'][window[:window_size-1]])
            avg_jerk = np.mean(scalars['jerk_mag'][window[:window_size-2]])
            
            features.append([avg_speed, max_accel, avg_jerk])
        
        features = np.array(features)
        
        # Apply weights if provided
        if weights:
            w = np.array([
                weights.get('speed', 1.0),
                weights.get('accel', 1.0),
                weights.get('jerk', 1.0)
            ])
            features = features * w
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return labels, features, kmeans.cluster_centers_
    
    def convert_to_physical_units(self, derivatives: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert pixel-based derivatives to physical units (meters, m/s, etc.)
        
        Args:
            derivatives: Dictionary with pixel-based derivatives
            
        Returns:
            Dictionary with physical unit derivatives
        """
        physical = {}
        
        # Velocity: pixels/s -> m/s
        physical['velocity_m_s'] = derivatives['velocity'] / self.pixels_per_meter
        
        # Acceleration: pixels/s² -> m/s²
        physical['acceleration_m_s2'] = derivatives['acceleration'] / self.pixels_per_meter
        
        # Jerk: pixels/s³ -> m/s³
        physical['jerk_m_s3'] = derivatives['jerk'] / self.pixels_per_meter
        
        # Jounce: pixels/s⁴ -> m/s⁴
        physical['jounce_m_s4'] = derivatives['jounce'] / self.pixels_per_meter
        
        return physical


class KalmanFilter:
    """Simple 2D Kalman Filter for tracking."""
    
    def __init__(self, dt: float, process_noise: float = 1.0, measurement_noise: float = 1.0):
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe only position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrices
        self.P = np.eye(4) * 100  # State covariance
        self.Q = np.eye(4) * process_noise  # Process noise
        self.R = np.eye(2) * measurement_noise  # Measurement noise
        
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """Update state with measurement."""
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[:2]


class FlockTracker(MotionAnalyzer):
    """Tracker for multiple objects (e.g., starling flock)"""
    
    def __init__(self, fps: float):
        super().__init__(fps)
        self.paths = []  # List of paths, one per bird
        self.n_objects = 0
        
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
    
    def associate_detections(self, prev_positions: List[np.ndarray], 
                            curr_centroids: List[Tuple[int, int]],
                            max_distance: float = 50.0) -> List[Optional[int]]:
        """
        Associate current detections with previous tracks using nearest neighbor.
        
        Args:
            prev_positions: List of previous positions (one per track)
            curr_centroids: List of current detections
            max_distance: Maximum association distance
            
        Returns:
            List mapping each previous track to current centroid index (or None)
        """
        if len(prev_positions) == 0 or len(curr_centroids) == 0:
            return [None] * len(prev_positions)
            
        # Compute distance matrix
        distances = np.zeros((len(prev_positions), len(curr_centroids)))
        for i, prev_pos in enumerate(prev_positions):
            for j, curr_pos in enumerate(curr_centroids):
                distances[i, j] = np.linalg.norm(prev_pos - np.array(curr_pos))
        
        # Greedy assignment (simple nearest neighbor)
        assignments = [None] * len(prev_positions)
        used_centroids = set()
        
        for i in range(len(prev_positions)):
            # Find nearest unused centroid
            valid_distances = distances[i].copy()
            for j in used_centroids:
                valid_distances[j] = float('inf')
                
            if valid_distances.min() < max_distance:
                j = valid_distances.argmin()
                assignments[i] = j
                used_centroids.add(j)
                
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
    
    def cluster_by_behavior(self, n_clusters: int = 2, 
                           weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Cluster objects by their motion behavior.
        
        Args:
            n_clusters: Number of clusters (e.g., leaders vs followers)
            weights: Weights for 'speed', 'accel', 'jerk'
            
        Returns:
            Array of cluster labels (one per object)
        """
        if len(self.paths) == 0:
            raise ValueError("No path data. Run track_video first.")
            
        # Compute feature vector for each object
        features = []
        for path in self.paths:
            if len(path) < 10:  # Skip very short tracks
                features.append([0, 0, 0])
                continue
                
            # Smooth and compute derivatives
            smooth_path = self.smooth_path(path)
            derivatives = self.compute_derivatives(smooth_path)
            scalars = self.compute_scalar_derivatives(derivatives)
            
            # Aggregate over entire trajectory
            avg_speed = np.mean(scalars['speed'])
            avg_accel = np.mean(scalars['accel_mag'])
            avg_jerk = np.mean(scalars['jerk_mag'])
            
            features.append([avg_speed, avg_accel, avg_jerk])
        
        features = np.array(features)
        
        # Apply weights
        if weights:
            w = np.array([
                weights.get('speed', 1.0),
                weights.get('accel', 1.0),
                weights.get('jerk', 1.0)
            ])
            features = features * w
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
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