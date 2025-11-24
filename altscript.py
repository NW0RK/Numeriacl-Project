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
    # Option 1: Single template
    # TEMPLATE_PATHS = ['mario_standing.png']
    
    # Option 2: Multiple templates (for different animations/states)
    TEMPLATE_PATHS = [
        "Screenshot2025-11-14152604.png",
        "Screenshot2025-11-14152525.png",
        "Screenshot2025-11-14152504.png",
        "Screenshot2025-11-14152425.png"
    ]
    
    VIDEO_PATH = "mario2.mp4"     # Path to gameplay video
    FPS = 60                                       # Video frame rate
    PIXELS_PER_METER = 32.0                       # Calibration: pixels per meter
    DETECTION_THRESHOLD = 0.4                     # Template matching threshold (0-1)
    # ======================================================
    
    print("=" * 70)
    print("PROBLEM A: 2D SPRITE TRACKER")
    print("=" * 70)
    
    # Load templates
    print(f"\n1. Loading {len(TEMPLATE_PATHS)} template(s):")
    templates = []
    for i, path in enumerate(TEMPLATE_PATHS):
        template = cv2.imread(path)
        if template is None:
            print(f"   ⚠ WARNING: Could not load template {i}: {path}")
            print("   Skipping this template...")
            continue
        templates.append(template)
        print(f"   ✓ Template {i}: {path} ({template.shape[1]}x{template.shape[0]} px)")
    
    if len(templates) == 0:
        print("\nERROR: No valid templates loaded!")
        print("Check your TEMPLATE_PATHS and make sure files exist.")
        return
    
    print(f"\n   Total templates loaded: {len(templates)}")
    
    # Initialize tracker
    print(f"\n2. Initializing tracker (FPS={FPS}, PPM={PIXELS_PER_METER})")
    tracker = SpriteTracker(fps=FPS, template=templates, pixels_per_meter=PIXELS_PER_METER)
    
    # Track through video
    print(f"\n3. Tracking sprite in video: {VIDEO_PATH}")
    print(f"   Detection threshold: {DETECTION_THRESHOLD}")
    
    # Check if video file exists
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"\nERROR: Video file not found: {VIDEO_PATH}")
        print("Please check the VIDEO_PATH in the configuration section.")
        return
    
    path = tracker.track_video(VIDEO_PATH, threshold=DETECTION_THRESHOLD)
    print(f"   ✓ Tracked {len(path)} frames")
    
    if len(path) == 0:
        print("\nERROR: No frames were tracked successfully.")
        print("Try lowering the DETECTION_THRESHOLD or checking your templates.")
        print("Aborting analysis.")
        return

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
    if len(scalars['speed']) > 0:
        print(f"   Average speed:        {np.mean(scalars['speed']):.2f} pixels/s")
        print(f"   Max speed:            {np.max(scalars['speed']):.2f} pixels/s")
        print(f"   Average acceleration: {np.mean(scalars['accel_mag']):.2f} pixels/s²")
        print(f"   Max jerk:             {np.max(scalars['jerk_mag']):.2f} pixels/s³")
    else:
        print("   (Not enough data to compute statistics)")
    
    # Convert to physical units
    physical = tracker.convert_to_physical_units(derivatives)
    physical_scalars = tracker.compute_scalar_derivatives(physical)
    
    print("\n6. Motion Statistics (Physical Units):")
    if len(physical_scalars['speed']) > 0:
        print(f"   Average speed:        {np.mean(physical_scalars['speed']):.2f} m/s")
        print(f"   Max speed:            {np.max(physical_scalars['speed']):.2f} m/s")
        print(f"   Average acceleration: {np.mean(physical_scalars['accel_mag']):.2f} m/s²")
    else:
        print("   (Not enough data to compute statistics)")
    
    # Cluster motion states
    print("\n7. Clustering motion states...")
    print("   Using standard L2 norm (no weights):")
    labels_std, features_std, centers_std = tracker.cluster_motion_states(
        window_size=20,
        n_clusters=3,
        weights=None
    )
    print(f"   ✓ Found {len(np.unique(labels_std))} clusters")
    # Use bincount safely - handle case where labels might not start at 0
    unique_labels = np.unique(labels_std)
    if len(unique_labels) > 0:
        max_label = np.max(unique_labels)
        counts = np.bincount(labels_std, minlength=max_label + 1)
        print(f"   Cluster distribution: {dict(zip(range(len(counts)), counts))}")
    else:
        print(f"   Cluster distribution: (empty)")
    
    print("\n   Using weighted norm (emphasize jerk for jumping):")
    labels_jerk, features_jerk, centers_jerk = tracker.cluster_motion_states(
        window_size=20,
        n_clusters=3,
        weights={'speed': 1.0, 'accel': 1.0, 'jerk': 10.0}
    )
    print(f"   ✓ Found {len(np.unique(labels_jerk))} clusters")
    unique_labels_jerk = np.unique(labels_jerk)
    if len(unique_labels_jerk) > 0:
        max_label_jerk = np.max(unique_labels_jerk)
        counts_jerk = np.bincount(labels_jerk, minlength=max_label_jerk + 1)
        print(f"   Cluster distribution: {dict(zip(range(len(counts_jerk)), counts_jerk))}")
    else:
        print(f"   Cluster distribution: (empty)")
    
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
    VIDEO_PATH = 'starlings2.mp4'          # Path to flock video
    FPS = 30                                       # Video frame rate
    BINARY_THRESHOLD = 50                          # Threshold for bird detection (0-255)
    MIN_BIRD_AREA = 5                             # Minimum contour area (pixels)
    MAX_BIRD_AREA = 500                           # Maximum contour area (pixels)
    USE_KALMAN = True                             # Use Kalman filtering for smoothing
    PIXELS_PER_METER_OVERRIDE = None              # Set to float to bypass auto-calibration
    STARLING_WINGSPAN_METERS = 0.35               # Reference size for auto-calibration
    # ======================================================
    
    print("=" * 70)
    print("PROBLEM B: STARLING FLOCK TRACKER")
    print("=" * 70)
    
    # Initialize tracker
    print(f"\n1. Initializing flock tracker (FPS={FPS})")
    tracker = FlockTracker(fps=FPS, pixels_per_meter=PIXELS_PER_METER_OVERRIDE)
    
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
    
    if tracker.pixels_per_meter is not None:
        print(f"   Using override scale: {tracker.pixels_per_meter:.2f} pixels/m")
    else:
        estimated_ppm = tracker.estimate_pixels_per_meter(
            frame,
            threshold=BINARY_THRESHOLD,
            min_area=MIN_BIRD_AREA,
            max_area=MAX_BIRD_AREA,
            reference_size_m=STARLING_WINGSPAN_METERS
        )
        if estimated_ppm:
            tracker.set_pixels_per_meter(estimated_ppm)
            print(f"   ✓ Auto-calibrated scale: {estimated_ppm:.2f} px/m (ref {STARLING_WINGSPAN_METERS} m wingspan)")
        else:
            print("   ⚠ Auto-calibration failed; using pixel units only.")
    
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
    
    metrics_pixels = tracker.aggregate_scalar_metrics()
    
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
    # Use bincount safely
    unique_labels_std = np.unique(labels_std)
    if len(unique_labels_std) > 0:
        max_label_std = np.max(unique_labels_std)
        counts_std = np.bincount(labels_std, minlength=max_label_std + 1)
        print(f"   ✓ Cluster distribution: {dict(zip(range(len(counts_std)), counts_std))}")
    else:
        print(f"   ✓ Cluster distribution: (empty)")
    
    print("\n   Using weighted norm (emphasize accel/jerk for leaders):")
    labels_weighted, features_weighted, centers_weighted = tracker.cluster_by_behavior(
        n_clusters=2,
        weights={'speed': 1.0, 'accel': 5.0, 'jerk': 5.0}
    )
    unique_labels_weighted = np.unique(labels_weighted)
    if len(unique_labels_weighted) > 0:
        max_label_weighted = np.max(unique_labels_weighted)
        counts_weighted = np.bincount(labels_weighted, minlength=max_label_weighted + 1)
        print(f"   ✓ Cluster distribution: {dict(zip(range(len(counts_weighted)), counts_weighted))}")
    else:
        print(f"   ✓ Cluster distribution: (empty)")
    
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
    if tracker.pixels_per_meter and metrics_pixels['speed'].size > 0:
        scale = tracker.pixels_per_meter
        speed_m = metrics_pixels['speed'] / scale
        accel_m = metrics_pixels['accel_mag'] / scale
        jerk_m = metrics_pixels['jerk_mag'] / scale
        
        print(f"   Scale: {scale:.2f} pixels per meter")
        if speed_m.size > 0:
            print(f"   Average speed:        {np.mean(speed_m):.2f} m/s")
            print(f"   Max speed:            {np.max(speed_m):.2f} m/s")
        if accel_m.size > 0:
            print(f"   Average acceleration: {np.mean(accel_m):.2f} m/s²")
            print(f"   Max acceleration:     {np.max(accel_m):.2f} m/s²")
        if jerk_m.size > 0:
            print(f"   Max jerk:             {np.max(jerk_m):.2f} m/s³")
    else:
        print("   ⚠ Could not convert to physical units (missing scale or motion samples)")
        if metrics_pixels['speed'].size > 0:
            print("   Pixel-unit fallback statistics:")
            print(f"     Avg speed: {np.mean(metrics_pixels['speed']):.2f} px/s")
            print(f"     Max speed: {np.max(metrics_pixels['speed']):.2f} px/s")
            print(f"     Avg accel: {np.mean(metrics_pixels['accel_mag']):.2f} px/s²")
    
    
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