import cv2
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# --- 1. Object Detection (MODIFIED) ---

def detect_sprite_template_matching(frame, templates_list):
    """
    Finds the sprite in the frame by trying a list of templates
    and returning the best match among all of them.
    
    Returns:
        (x, y) coordinate of the center of the best match, and its score.
    """
    if frame is None or not templates_list:
        return None

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    best_match = {
        'score': -1, # Initialize with a score lower than any possible match
        'position': (0, 0),
        'template_shape': (0, 0)
    }

    # Loop through each template to find the best one
    for template in templates_list:
        if template is None:
            continue
            
        h, w = template.shape[:2]
        if h > frame_gray.shape[0] or w > frame_gray.shape[1]:
            # Skip template if it's larger than the frame
            continue
            
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # If this template's best score is better than our overall best, update it
        if max_val > best_match['score']:
            best_match['score'] = max_val
            best_match['position'] = max_loc # This is the top-left corner
            best_match['template_shape'] = (h, w)
            
    # If no match was found (score is still -1), return None
    if best_match['score'] == -1:
        return None, 0.0

    # Calculate the center of the *best* match
    top_left = best_match['position']
    h, w = best_match['template_shape']
    
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2
    
    return (center_x, center_y), best_match['score']

# --- 2. Path, Smoothing, and Derivatives ---

def calculate_derivatives(smooth_path, fps):
    """
    Calculates 1st to 4th order derivatives of a 2D path.
    """
    dt = 1.0 / fps
    p = smooth_path
    
    v = np.diff(p, n=1, axis=0) / dt
    v = np.pad(v, ((1, 0), (0, 0)), 'edge')
    
    a = np.diff(v, n=1, axis=0) / dt
    a = np.pad(a, ((1, 0), (0, 0)), 'edge')
    
    j = np.diff(a, n=1, axis=0) / dt
    j = np.pad(j, ((1, 0), (0, 0)), 'edge')
    
    s = np.diff(j, n=1, axis=0) / dt
    s = np.pad(s, ((1, 0), (0, 0)), 'edge')
    
    return p, v, a, j, s

# --- 3. Clustering by Motion ---

def cluster_motion_states(derivatives, window_size=20, n_clusters=3, weights=None):
    """
    Clusters the sprite's motion states over time using K-Means.
    """
    p, v, a, j, s = derivatives
    
    speed = np.linalg.norm(v, axis=1)
    acceleration = np.linalg.norm(a, axis=1)
    jerk = np.linalg.norm(j, axis=1)
    
    feature_vectors = []
    num_frames = len(p)
    
    for i in range(num_frames - window_size):
        segment_speed = speed[i:i + window_size]
        segment_accel = acceleration[i:i + window_size]
        segment_jerk = jerk[i:i + window_size]
        
        vec = [
            np.mean(segment_speed),
            np.max(segment_accel),
            np.mean(segment_jerk)
        ]
        feature_vectors.append(vec)
        
    if not feature_vectors:
        print("Warning: No feature vectors created. Video might be too short.")
        return None, None

    features = np.array(feature_vectors)
    
    features_scaled = features.copy()
    if weights:
        if 'speed' in weights:
            features_scaled[:, 0] *= np.sqrt(weights.get('speed', 1))
        if 'accel' in weights:
            features_scaled[:, 1] *= np.sqrt(weights.get('accel', 1))
        if 'jerk' in weights:
            features_scaled[:, 2] *= np.sqrt(weights.get('jerk', 1))
            
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    
    full_labels = np.full(num_frames, -1)
    for i, label in enumerate(labels):
        full_labels[i] = label
        
    return full_labels, kmeans.cluster_centers_

# --- 4. Conversion to Physical Units ---

def convert_to_metric(pixel_values, ppm):
    """
    Converts values from pixel units to metric units.
    """
    return pixel_values / ppm

# --- Main Driver Function (MODIFIED) ---

def run_mario_tracker(video_path, template_paths_list, fps=60, ppm=32, match_threshold=0.8):
    """
    MODIFIED: This function now accepts a LIST of template paths.
    """
    
    print("--- Starting Problem A: Mario Tracker ---")
    
    # --- MODIFIED SECTION: Load all templates ---
    print(f"Loading {len(template_paths_list)} templates...")
    templates = []
    for path in template_paths_list:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Warning: Template image not found at {path}. Skipping it.")
        else:
            templates.append(template)
            
    if not templates:
        raise FileNotFoundError("No valid template images were loaded. Check paths.")
    print(f"Successfully loaded {len(templates)} templates.")
    # --- END MODIFIED SECTION ---

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found at {video_path}")
        
    raw_path_list = []
    
    print("Step 1: Detecting sprite (Multi-Template Matching)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- MODIFIED SECTION: Call detection with the list of templates ---
        result = detect_sprite_template_matching(frame, templates)
        # --- END MODIFIED SECTION ---
        
        if result:
            pos, score = result
            if score > match_threshold:
                raw_path_list.append(pos)
            else:
                if raw_path_list:
                    raw_path_list.append(raw_path_list[-1])
                else:
                    raw_path_list.append((0, 0))
        else:
            if raw_path_list:
                raw_path_list.append(raw_path_list[-1])
            else:
                raw_path_list.append((0, 0))
                
    cap.release()
    raw_path = np.array(raw_path_list)
    
    if len(raw_path) == 0:
        print("Error: No sprite positions were tracked.")
        return

    print(f"Tracking complete. Found path with {len(raw_path)} frames.")
    
    print("Step 2: Smoothing and calculating derivatives...")
    window_len = min(51, len(raw_path) - 1 if len(raw_path) % 2 == 0 else len(raw_path) - 2)
    
    if window_len < 5:
        print("Warning: Not enough data to smooth path. Using raw path.")
        smooth_path = raw_path
    else:
        smooth_x = savgol_filter(raw_path[:, 0], window_len, 3)
        smooth_y = savgol_filter(raw_path[:, 1], window_len, 3)
        smooth_path = np.stack([smooth_x, smooth_y], axis=1)

    derivatives = calculate_derivatives(smooth_path, fps)
    
    print("Step 3: Clustering motion states...")
    weights = {'speed': 1, 'accel': 1, 'jerk': 10}
    labels, centers = cluster_motion_states(derivatives, window_size=30, n_clusters=3, weights=weights)
    
    if labels is not None:
        print(f"Clustering complete. Found {len(centers)} clusters (states).")
    else:
        print("Clustering failed or was skipped.")
    
    print("Step 4: Converting units...")
    speed_pixels_per_sec = np.linalg.norm(derivatives[1], axis=1)
    speed_meters_per_sec = convert_to_metric(speed_pixels_per_sec, ppm)
    
    print("--- Analysis Complete ---")
    print(f"Max Speed (pixels/s): {np.max(speed_pixels_per_sec):.2f}")
    print(f"Max Speed (m/s): {np.max(speed_meters_per_sec):.2f}")
    
    if labels is not None:
        print(f"Motion states found: {np.unique(labels[labels != -1])}")

# --- Example Usage (MODIFIED) ---

if __name__ == "__main__":
    
    VIDEO_FILE = "C:/Users/niko0/Downloads/output.mp4" # 1. Replace with your video file
    
    # 2. Replace with your four template images
    TEMPLATE_FILES_LIST = [
        'c:/Users/niko0/Pictures/Screenshots/Screenshot2025-11-14152604.png',
        'c:/Users/niko0/Pictures/Screenshots/Screenshot2025-11-14152525.png',
        'c:/Users/niko0/Pictures/Screenshots/Screenshot2025-11-14152504.png',
        'c:/Users/niko0/Pictures/Screenshots/Screenshot2025-11-14152425.png'
    ]
    
    # 
    
    # --- !! TUNE THESE !! ---
    VIDEO_FPS = 60
    PIXELS_PER_METER = 32
    MATCH_CONFIDENCE = 0.5 # 80% confidence threshold

    try:
        # Pass the list of file paths to the tracker
        run_mario_tracker(
            VIDEO_FILE,
            TEMPLATE_FILES_LIST,
            fps=VIDEO_FPS,
            ppm=PIXELS_PER_METER,
            match_threshold=MATCH_CONFIDENCE
        )
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")