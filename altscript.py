"""
Usage script for Motion Tracking Implementation
Run this script after saving the main implementation as 'motion_tracker.py'
"""

import cv2
import numpy as np
from motion_tracker import SpriteTracker, FlockTracker, visualize_sprite_tracking, visualize_flock_tracking
import matplotlib.pyplot as plt


def format_cluster_distribution(counts: np.ndarray) -> dict:
    """Return dict of cluster counts while preserving numpy scalar repr."""
    if counts is None:
        return {}
    return {int(idx): counts[idx] for idx in range(len(counts))}


def save_console_snapshot(title: str, sections, output_path: str):
    """
    Render console-style text blocks into an image for reporting.
    
    Args:
        title: Figure title text.
        sections: Iterable of (header, lines) pairs where lines is a list of strings.
        output_path: File path to save the rendered figure.
    """
    total_lines = sum(len(lines) + 2 for _, lines in sections) + 2
    fig_height = max(4, 0.28 * total_lines)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    
    y = 0.98
    ax.text(0.02, y, title, fontsize=16, fontweight='bold',
            va='top', family='monospace')
    y -= 0.06
    line_step = 0.035
    
    for header, lines in sections:
        block_lines = [header] + lines
        ax.text(0.02, y, "\n".join(block_lines), fontsize=11,
                va='top', family='monospace')
        y -= line_step * (len(block_lines) + 1)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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
    pixel_stats = None
    print("\n5. Motion Statistics (Pixel Units):")
    if len(scalars['speed']) > 0:
        pixel_stats = {
            'avg_speed': np.mean(scalars['speed']),
            'max_speed': np.max(scalars['speed']),
            'avg_accel': np.mean(scalars['accel_mag']),
            'max_jerk': np.max(scalars['jerk_mag'])
        }
        print(f"   Average speed:        {pixel_stats['avg_speed']:.2f} pixels/s")
        print(f"   Max speed:            {pixel_stats['max_speed']:.2f} pixels/s")
        print(f"   Average acceleration: {pixel_stats['avg_accel']:.2f} pixels/s²")
        print(f"   Max jerk:             {pixel_stats['max_jerk']:.2f} pixels/s³")
    else:
        print("   (Not enough data to compute statistics)")
    
    # Convert to physical units
    physical = tracker.convert_to_physical_units(derivatives)
    physical_scalars = tracker.compute_scalar_derivatives(physical)
    
    physical_stats = None
    print("\n6. Motion Statistics (Physical Units):")
    if len(physical_scalars['speed']) > 0:
        physical_stats = {
            'avg_speed': np.mean(physical_scalars['speed']),
            'max_speed': np.max(physical_scalars['speed']),
            'avg_accel': np.mean(physical_scalars['accel_mag'])
        }
        print(f"   Average speed:        {physical_stats['avg_speed']:.2f} m/s")
        print(f"   Max speed:            {physical_stats['max_speed']:.2f} m/s")
        print(f"   Average acceleration: {physical_stats['avg_accel']:.2f} m/s²")
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
    cluster_dist_std = None
    clusters_found_std = len(np.unique(labels_std))
    if len(unique_labels) > 0:
        max_label = np.max(unique_labels)
        counts = np.bincount(labels_std, minlength=max_label + 1)
        print(f"   Cluster distribution: {dict(zip(range(len(counts)), counts))}")
        cluster_dist_std = format_cluster_distribution(counts)
    else:
        print(f"   Cluster distribution: (empty)")
    
    print("\n   Using weighted norm (emphasize jerk for jumping):")
    labels_jerk, features_jerk, centers_jerk = tracker.cluster_motion_states(
        window_size=20,
        n_clusters=3,
        weights={'speed': 1.0, 'accel': 1.0, 'jerk': 10.0}
    )
    clusters_found_weighted = len(np.unique(labels_jerk))
    print(f"   ✓ Found {clusters_found_weighted} clusters")
    unique_labels_jerk = np.unique(labels_jerk)
    if len(unique_labels_jerk) > 0:
        max_label_jerk = np.max(unique_labels_jerk)
        counts_jerk = np.bincount(labels_jerk, minlength=max_label_jerk + 1)
        print(f"   Cluster distribution: {dict(zip(range(len(counts_jerk)), counts_jerk))}")
        cluster_dist_weighted = format_cluster_distribution(counts_jerk)
    else:
        print(f"   Cluster distribution: (empty)")
        cluster_dist_weighted = None
    
    # Visualize
    print("\n8. Generating visualizations...")
    fig = visualize_sprite_tracking(tracker)
    plt.savefig('sprite_tracking_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: sprite_tracking_results.png")
    plt.show()
    
    summary_sections = [
        ("5. Motion Statistics (Pixel Units):", [
            f"   Average speed:        {pixel_stats['avg_speed']:.2f} pixels/s",
            f"   Max speed:            {pixel_stats['max_speed']:.2f} pixels/s",
            f"   Average acceleration: {pixel_stats['avg_accel']:.2f} pixels/s²",
            f"   Max jerk:             {pixel_stats['max_jerk']:.2f} pixels/s³",
        ] if pixel_stats else ["   (Not enough data to compute statistics)"]),
        ("6. Motion Statistics (Physical Units):", [
            f"   Average speed:        {physical_stats['avg_speed']:.2f} m/s",
            f"   Max speed:            {physical_stats['max_speed']:.2f} m/s",
            f"   Average acceleration: {physical_stats['avg_accel']:.2f} m/s²",
        ] if physical_stats else ["   (Not enough data to compute statistics)"]),
        ("7. Clustering motion states...", [
            "   Using standard L2 norm (no weights):",
            f"   ✓ Found {clusters_found_std} clusters",
            f"   Cluster distribution: {cluster_dist_std if cluster_dist_std is not None else '(empty)'}",
            "",
            "   Using weighted norm (emphasize jerk for jumping):",
            f"   ✓ Found {clusters_found_weighted} clusters",
            f"   Cluster distribution: {cluster_dist_weighted if cluster_dist_weighted is not None else '(empty)'}",
        ])
    ]
    save_console_snapshot(
        "Sprite Tracker Console Summary",
        summary_sections,
        "sprite_tracking_console.png"
    )
    print("   ✓ Saved: sprite_tracking_console.png (console summary)")
    
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
    cluster_dist_std = None
    clusters_found_std = len(np.unique(labels_std))
    if len(unique_labels_std) > 0:
        max_label_std = np.max(unique_labels_std)
        counts_std = np.bincount(labels_std, minlength=max_label_std + 1)
        cluster_dist_std = format_cluster_distribution(counts_std)
        print(f"   ✓ Found {clusters_found_std} clusters")
        print(f"   ✓ Cluster distribution: {cluster_dist_std}")
    else:
        print(f"   ✓ Found 0 clusters")
        print(f"   ✓ Cluster distribution: (empty)")
        cluster_dist_std = None
    
    print("\n   Using weighted norm (emphasize accel/jerk for leaders):")
    labels_weighted, features_weighted, centers_weighted = tracker.cluster_by_behavior(
        n_clusters=2,
        weights={'speed': 1.0, 'accel': 5.0, 'jerk': 5.0}
    )
    unique_labels_weighted = np.unique(labels_weighted)
    cluster_dist_weighted = None
    clusters_found_weighted = len(np.unique(labels_weighted))
    if len(unique_labels_weighted) > 0:
        max_label_weighted = np.max(unique_labels_weighted)
        counts_weighted = np.bincount(labels_weighted, minlength=max_label_weighted + 1)
        cluster_dist_weighted = format_cluster_distribution(counts_weighted)
        print(f"   ✓ Found {clusters_found_weighted} clusters")
        print(f"   ✓ Cluster distribution: {cluster_dist_weighted}")
    else:
        print(f"   ✓ Found 0 clusters")
        print(f"   ✓ Cluster distribution: (empty)")
        cluster_dist_weighted = None
    
    # Analyze clusters
    print("\n4. Cluster Analysis:")
    cluster_details = []
    for cluster_id in range(len(cluster_dist_weighted) if cluster_dist_weighted else 2):
        mask = labels_weighted == cluster_id
        cluster_features = features_weighted[mask]
        count = int(np.sum(mask))
        if count > 0:
            avg_speed = np.mean(cluster_features[:, 0])
            avg_accel = np.mean(cluster_features[:, 1])
            avg_jerk = np.mean(cluster_features[:, 2])
        else:
            avg_speed = avg_accel = avg_jerk = float('nan')
        cluster_details.append({
            'id': cluster_id,
            'count': count,
            'avg_speed': avg_speed,
            'avg_accel': avg_accel,
            'avg_jerk': avg_jerk
        })
        print(f"\n   Cluster {cluster_id} ({count} birds):")
        print(f"     Avg speed:        {avg_speed:.2f}")
        print(f"     Avg acceleration: {avg_accel:.2f}")
        print(f"     Avg jerk:         {avg_jerk:.2f}")
    
    # Visualize
    print("\n5. Generating visualizations...")
    fig = visualize_flock_tracking(tracker, cluster_labels=labels_weighted)
    plt.savefig('flock_tracking_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: flock_tracking_results.png")
    plt.show()
    
    print("\n6. Physical Units:")
    physical_report = {
        'available': False,
        'scale': None,
        'avg_speed': None,
        'max_speed': None,
        'avg_accel': None,
        'max_accel': None,
        'max_jerk': None,
        'fallback': None
    }
    if tracker.pixels_per_meter and metrics_pixels['speed'].size > 0:
        scale = tracker.pixels_per_meter
        speed_m = metrics_pixels['speed'] / scale
        accel_m = metrics_pixels['accel_mag'] / scale
        jerk_m = metrics_pixels['jerk_mag'] / scale
        
        physical_report.update({
            'available': True,
            'scale': scale,
            'avg_speed': np.mean(speed_m) if speed_m.size > 0 else None,
            'max_speed': np.max(speed_m) if speed_m.size > 0 else None,
            'avg_accel': np.mean(accel_m) if accel_m.size > 0 else None,
            'max_accel': np.max(accel_m) if accel_m.size > 0 else None,
            'max_jerk': np.max(jerk_m) if jerk_m.size > 0 else None
        })
        
        print(f"   Scale: {scale:.2f} pixels per meter")
        if speed_m.size > 0:
            print(f"   Average speed:        {physical_report['avg_speed']:.2f} m/s")
            print(f"   Max speed:            {physical_report['max_speed']:.2f} m/s")
        if accel_m.size > 0:
            print(f"   Average acceleration: {physical_report['avg_accel']:.2f} m/s²")
            print(f"   Max acceleration:     {physical_report['max_accel']:.2f} m/s²")
        if jerk_m.size > 0:
            print(f"   Max jerk:             {physical_report['max_jerk']:.2f} m/s³")
    else:
        print("   ⚠ Could not convert to physical units (missing scale or motion samples)")
        if metrics_pixels['speed'].size > 0:
            fallback = {
                'avg_speed': np.mean(metrics_pixels['speed']),
                'max_speed': np.max(metrics_pixels['speed']),
                'avg_accel': np.mean(metrics_pixels['accel_mag'])
            }
            physical_report['fallback'] = fallback
            print("   Pixel-unit fallback statistics:")
            print(f"     Avg speed: {fallback['avg_speed']:.2f} px/s")
            print(f"     Max speed: {fallback['max_speed']:.2f} px/s")
            print(f"     Avg accel: {fallback['avg_accel']:.2f} px/s²")
    
    clustering_lines = [
        "   Using standard L2 norm:",
        f"   ✓ Found {clusters_found_std} clusters",
        f"   ✓ Cluster distribution: {cluster_dist_std if cluster_dist_std is not None else '(empty)'}",
        "",
        "   Using weighted norm (emphasize accel/jerk for leaders):",
        f"   ✓ Found {clusters_found_weighted} clusters",
        f"   ✓ Cluster distribution: {cluster_dist_weighted if cluster_dist_weighted is not None else '(empty)'}"
    ]
    
    cluster_analysis_lines = []
    if cluster_details:
        for detail in cluster_details:
            cluster_analysis_lines.extend([
                f"   Cluster {detail['id']} ({detail['count']} birds):",
                f"     Avg speed:        {detail['avg_speed']:.2f}",
                f"     Avg acceleration: {detail['avg_accel']:.2f}",
                f"     Avg jerk:         {detail['avg_jerk']:.2f}",
                ""
            ])
        if cluster_analysis_lines:
            cluster_analysis_lines.pop()  # remove last blank line
    else:
        cluster_analysis_lines.append("   (No cluster statistics available)")
    
    if physical_report['available']:
        physical_lines = [f"   Scale: {physical_report['scale']:.2f} pixels per meter"]
        if physical_report['avg_speed'] is not None:
            physical_lines.append(f"   Average speed:        {physical_report['avg_speed']:.2f} m/s")
        if physical_report['max_speed'] is not None:
            physical_lines.append(f"   Max speed:            {physical_report['max_speed']:.2f} m/s")
        if physical_report['avg_accel'] is not None:
            physical_lines.append(f"   Average acceleration: {physical_report['avg_accel']:.2f} m/s²")
        if physical_report['max_accel'] is not None:
            physical_lines.append(f"   Max acceleration:     {physical_report['max_accel']:.2f} m/s²")
        if physical_report['max_jerk'] is not None:
            physical_lines.append(f"   Max jerk:             {physical_report['max_jerk']:.2f} m/s³")
    elif physical_report['fallback']:
        physical_lines = [
            "   ⚠ Could not convert to physical units.",
            "   Pixel-unit fallback statistics:",
            f"     Avg speed: {physical_report['fallback']['avg_speed']:.2f} px/s",
            f"     Max speed: {physical_report['fallback']['max_speed']:.2f} px/s",
            f"     Avg accel: {physical_report['fallback']['avg_accel']:.2f} px/s²",
        ]
    else:
        physical_lines = ["   ⚠ Could not convert to physical units (no motion samples)."]
    
    summary_sections = [
        ("3. Clustering by motion behavior...", clustering_lines),
        ("4. Cluster Analysis:", cluster_analysis_lines),
        ("6. Physical Units:", physical_lines)
    ]
    save_console_snapshot(
        "Flock Tracker Console Summary",
        summary_sections,
        "flock_tracking_console.png"
    )
    print("   ✓ Saved: flock_tracking_console.png (console summary)")
    
    
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