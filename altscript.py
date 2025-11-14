"""
Usage script for Motion Tracking Implementation
Run this script after saving the main implementation as 'motion_tracker.py'
"""

import cv2
import numpy as np
from motion_tracker import SpriteTracker, FlockTracker, visualize_sprite_tracking, visualize_flock_tracking
import matplotlib.pyplot as plt

# ==============================================================================
# PROBLEM A: 2D SPRITE TRACKER (Super Mario Bros.)
# ==============================================================================

def run_sprite_tracker():
    """Track a single sprite through video."""
    
    # ========== CONFIGURATION - EDIT THESE PATHS ==========
    TEMPLATE_PATH = 'c:/Users/Nikoloz/Pictures/Screenshots/Screenshot2025-11-14152604.png'  # Path to sprite template image
    VIDEO_PATH = 'C:/Users/Nikoloz/Downloads/output.mp4'     # Path to gameplay video
    FPS = 60                                       # Video frame rate
    PIXELS_PER_METER = 32.0                       # Calibration: pixels per meter
    DETECTION_THRESHOLD = 0.8                     # Template matching threshold (0-1)
    # ======================================================
    
    print("=" * 70)
    print("PROBLEM A: 2D SPRITE TRACKER")
    print("=" * 70)
    
    # Load template
    print(f"\n1. Loading template from: {TEMPLATE_PATH}")
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print(f"ERROR: Could not load template from {TEMPLATE_PATH}")
        print("Make sure the path is correct and the file exists.")
        return
    print(f"   Template size: {template.shape[1]}x{template.shape[0]} pixels")
    
    # Initialize tracker
    print(f"\n2. Initializing tracker (FPS={FPS}, PPM={PIXELS_PER_METER})")
    tracker = SpriteTracker(fps=FPS, template=template, pixels_per_meter=PIXELS_PER_METER)
    
    # Track through video
    print(f"\n3. Tracking sprite in video: {VIDEO_PATH}")
    print(f"   Detection threshold: {DETECTION_THRESHOLD}")
    path = tracker.track_video(VIDEO_PATH, threshold=DETECTION_THRESHOLD)
    print(f"   ✓ Tracked {len(path)} frames")
    
    # Smooth path and compute derivatives
    print("\n4. Computing derivatives...")
    smooth_path = tracker.smooth_path(path)
    derivatives = tracker.compute_derivatives(smooth_path)
    scalars = tracker.compute_scalar_derivatives(derivatives)
    
    print(f"   ✓ Velocity:     {len(derivatives['velocity'])} samples")
    print(f"   ✓ Acceleration: {len(derivatives['acceleration'])} samples")
    print(f"   ✓ Jerk:         {len(derivatives['jerk'])} samples")
    print(f"   ✓ Jounce:       {len(derivatives['jounce'])} samples")
    
    # Statistics
    print("\n5. Motion Statistics (Pixel Units):")
    print(f"   Average speed:        {np.mean(scalars['speed']):.2f} pixels/s")
    print(f"   Max speed:            {np.max(scalars['speed']):.2f} pixels/s")
    print(f"   Average acceleration: {np.mean(scalars['accel_mag']):.2f} pixels/s²")
    print(f"   Max jerk:             {np.max(scalars['jerk_mag']):.2f} pixels/s³")
    
    # Convert to physical units
    physical = tracker.convert_to_physical_units(derivatives)
    physical_scalars = tracker.compute_scalar_derivatives(physical)
    
    print("\n6. Motion Statistics (Physical Units):")
    print(f"   Average speed:        {np.mean(physical_scalars['speed']):.2f} m/s")
    print(f"   Max speed:            {np.max(physical_scalars['speed']):.2f} m/s")
    print(f"   Average acceleration: {np.mean(physical_scalars['accel_mag']):.2f} m/s²")
    
    # Cluster motion states
    print("\n7. Clustering motion states...")
    print("   Using standard L2 norm (no weights):")
    labels_std, features_std, centers_std = tracker.cluster_motion_states(
        window_size=20,
        n_clusters=3,
        weights=None
    )
    print(f"   ✓ Found {len(np.unique(labels_std))} clusters")
    print(f"   Cluster distribution: {np.bincount(labels_std)}")
    
    print("\n   Using weighted norm (emphasize jerk for jumping):")
    labels_jerk, features_jerk, centers_jerk = tracker.cluster_motion_states(
        window_size=20,
        n_clusters=3,
        weights={'speed': 1.0, 'accel': 1.0, 'jerk': 10.0}
    )
    print(f"   ✓ Found {len(np.unique(labels_jerk))} clusters")
    print(f"   Cluster distribution: {np.bincount(labels_jerk)}")
    
    # Visualize
    print("\n8. Generating visualizations...")
    fig = visualize_sprite_tracking(tracker)
    plt.savefig('sprite_tracking_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sprite_tracking_results.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("SPRITE TRACKING COMPLETE")
    print("=" * 70)


# ==============================================================================
# PROBLEM B: STARLING FLOCK TRACKER
# ==============================================================================

def run_flock_tracker():
    """Track multiple objects (birds) through video."""
    
    # ========== CONFIGURATION - EDIT THESE PATHS ==========
    VIDEO_PATH = 'path/to/starlings.mp4'          # Path to flock video
    FPS = 30                                       # Video frame rate
    BINARY_THRESHOLD = 50                          # Threshold for bird detection (0-255)
    MIN_BIRD_AREA = 5                             # Minimum contour area (pixels)
    MAX_BIRD_AREA = 500                           # Maximum contour area (pixels)
    USE_KALMAN = True                             # Use Kalman filtering for smoothing
    # ======================================================
    
    print("=" * 70)
    print("PROBLEM B: STARLING FLOCK TRACKER")
    print("=" * 70)
    
    # Initialize tracker
    print(f"\n1. Initializing flock tracker (FPS={FPS})")
    tracker = FlockTracker(fps=FPS)
    
    # Track through video
    print(f"\n2. Tracking flock in video: {VIDEO_PATH}")
    print(f"   Binary threshold:     {BINARY_THRESHOLD}")
    print(f"   Bird area range:      {MIN_BIRD_AREA}-{MAX_BIRD_AREA} pixels")
    print(f"   Kalman filtering:     {'Enabled' if USE_KALMAN else 'Disabled'}")
    
    # First, check how many objects are detected
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"ERROR: Could not read video from {VIDEO_PATH}")
        print("Make sure the path is correct and the file exists.")
        return
    
    test_centroids = tracker.detect_objects(frame, threshold=BINARY_THRESHOLD, 
                                           min_area=MIN_BIRD_AREA, max_area=MAX_BIRD_AREA)
    print(f"   Detected {len(test_centroids)} objects in first frame")
    
    if len(test_centroids) == 0:
        print("\n   WARNING: No objects detected! Try adjusting:")
        print("   - BINARY_THRESHOLD (lower for darker objects, higher for lighter)")
        print("   - MIN_BIRD_AREA and MAX_BIRD_AREA")
        return
    
    # Full tracking
    paths = tracker.track_video(VIDEO_PATH, threshold=BINARY_THRESHOLD, use_kalman=USE_KALMAN)
    print(f"   ✓ Tracked {len(paths)} objects")
    
    if len(paths) == 0:
        print("   ERROR: No objects were tracked successfully")
        return
    
    # Statistics
    path_lengths = [len(p) for p in paths]
    print(f"   Average track length: {np.mean(path_lengths):.1f} frames")
    print(f"   Shortest track:       {np.min(path_lengths)} frames")
    print(f"   Longest track:        {np.max(path_lengths)} frames")
    
    # Cluster by behavior
    print("\n3. Clustering by motion behavior...")
    print("   Using standard L2 norm:")
    labels_std, features_std, centers_std = tracker.cluster_by_behavior(
        n_clusters=2,
        weights=None
    )
    print(f"   ✓ Cluster distribution: {np.bincount(labels_std)}")
    
    print("\n   Using weighted norm (emphasize accel/jerk for leaders):")
    labels_weighted, features_weighted, centers_weighted = tracker.cluster_by_behavior(
        n_clusters=2,
        weights={'speed': 1.0, 'accel': 5.0, 'jerk': 5.0}
    )
    print(f"   ✓ Cluster distribution: {np.bincount(labels_weighted)}")
    
    # Analyze clusters
    print("\n4. Cluster Analysis:")
    for cluster_id in range(2):
        mask = labels_weighted == cluster_id
        cluster_features = features_weighted[mask]
        print(f"\n   Cluster {cluster_id} ({np.sum(mask)} birds):")
        print(f"     Avg speed:        {np.mean(cluster_features[:, 0]):.2f}")
        print(f"     Avg acceleration: {np.mean(cluster_features[:, 1]):.2f}")
        print(f"     Avg jerk:         {np.mean(cluster_features[:, 2]):.2f}")
    
    # Visualize
    print("\n5. Generating visualizations...")
    fig = visualize_flock_tracking(tracker, cluster_labels=labels_weighted)
    plt.savefig('flock_tracking_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: flock_tracking_results.png")
    plt.show()
    
    print("\n6. Physical Units:")
    print("   ⚠ Cannot convert to physical units (3D perspective)")
    print("   All measurements remain in pixel-based units")
    
    print("\n" + "=" * 70)
    print("FLOCK TRACKING COMPLETE")
    print("=" * 70)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "MOTION TRACKER USAGE" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\nSelect tracking mode:")
    print("1. Sprite Tracker (Problem A - Super Mario)")
    print("2. Flock Tracker (Problem B - Starlings)")
    print("3. Run both")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        run_sprite_tracker()
    elif choice == '2':
        run_flock_tracker()
    elif choice == '3':
        print("\n" + ">" * 70)
        run_sprite_tracker()
        print("\n" + ">" * 70 + "\n")
        run_flock_tracker()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice!")
        
    print("\n✓ Done!\n")