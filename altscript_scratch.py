"""
Alternate launcher that uses the scratch-built motion tracker implementation.

This script mirrors the behavior of `altscript.py` but routes all analysis
through `motion_tracker_scratch.py`, which provides custom numerical helpers
instead of NumPy/SciPy/SKLearn.
"""

import os
from typing import List

import cv2
import matplotlib.pyplot as plt

from motion_tracker_scratch import (
    FlockTrackerScratch,
    SpriteTrackerScratch,
    visualize_flock_tracking,
    visualize_sprite_tracking,
)


# =============================================================================
# PROBLEM A: 2D SPRITE TRACKER (Scratch build)
# =============================================================================

def run_sprite_tracker():
    template_paths: List[str] = [
        "Screenshot2025-11-14152604.png",
        "Screenshot2025-11-14152525.png",
        "Screenshot2025-11-14152504.png",
        "Screenshot2025-11-14152425.png",
    ]
    video_path = "mario2.mp4"
    fps = 60.0
    ppm = 32.0
    threshold = 0.35

    print("=" * 70)
    print("SCRATCH MODE: SPRITE TRACKER")
    print("=" * 70)

    templates = []
    for path in template_paths:
        tmpl = cv2.imread(path)
        if tmpl is None:
            print(f"   ⚠ Could not load template: {path}")
            continue
        templates.append(tmpl)
    if not templates:
        print("No templates loaded; aborting.")
        return
    if not os.path.exists(video_path):
        print(f"Video missing: {video_path}")
        return

    tracker = SpriteTrackerScratch(fps=fps, templates=templates, pixels_per_meter=ppm)
    path = tracker.track_video(video_path, threshold=threshold)
    smooth = tracker.smooth_path(path)
    derivatives = tracker.compute_derivatives(smooth)
    scalars = tracker.compute_scalar_derivatives(derivatives)
    physical = tracker.convert_to_physical_units(derivatives)
    physical_scalars = tracker.compute_scalar_derivatives(physical)

    print(f"Tracked {len(path)} frames.")
    if scalars["speed"]:
        print(f"Average speed (px/s): {_avg(scalars['speed']):.2f}")
        print(f"Max jerk (px/s^3):    {max(scalars['jerk_mag']):.2f}")
    if physical_scalars["speed"]:
        print(f"Average speed (m/s): {_avg(physical_scalars['speed']):.2f}")

    labels, features, centers = tracker.cluster_motion_states(window_size=20, n_clusters=3)
    print(f"Cluster counts: {_hist(labels)}")

    viz = visualize_sprite_tracking(tracker)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    raw = viz["path"]
    smooth_path = viz["smooth_path"]
    axes[0, 0].plot([p[0] for p in raw], [p[1] for p in raw], "b.", alpha=0.3, label="Raw")
    axes[0, 0].plot([p[0] for p in smooth_path], [p[1] for p in smooth_path], "r-", label="Smooth")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()
    axes[0, 0].set_title("Sprite Path")

    scalars = viz["scalars"]
    axes[0, 1].plot(scalars["speed"])
    axes[0, 1].set_title("Speed")
    axes[1, 0].plot(scalars["accel_mag"])
    axes[1, 0].set_title("Acceleration")
    axes[1, 1].plot(scalars["jerk_mag"])
    axes[1, 1].set_title("Jerk")
    plt.tight_layout()
    plt.savefig("sprite_tracking_scratch.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PROBLEM B: FLOCK TRACKER (Scratch build)
# =============================================================================

def run_flock_tracker():
    video_path = "starlings2.mp4"
    fps = 30.0
    threshold = 60
    min_area = 5
    max_area = 500

    print("=" * 70)
    print("SCRATCH MODE: FLOCK TRACKER")
    print("=" * 70)

    tracker = FlockTrackerScratch(fps=fps)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read first frame from {video_path}")
        return

    detections = tracker.detect_objects(frame, threshold, min_area, max_area)
    print(f"Detected {len(detections)} birds in first frame.")
    if not detections:
        print("Adjust threshold or area bounds.")
        return

    paths = tracker.track_video(video_path, threshold=threshold, use_kalman=True)
    print(f"Tracked {len(paths)} birds.")

    labels, features, centers = tracker.cluster_by_behavior(n_clusters=2)
    print(f"Behavior clusters: {_hist(labels)}")

    viz = visualize_flock_tracking(tracker, labels)
    plt.figure(figsize=(10, 6))
    for idx, path in enumerate(viz["paths"]):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, alpha=0.6, label=f"Bird {idx} (C{labels[idx]})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("flock_tracking_scratch.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Utilities & CLI
# =============================================================================

def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _hist(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    print("\nScratch-built Motion Tracker Demo\n")
    print("1. Sprite tracker")
    print("2. Flock tracker")
    print("3. Run both")
    print("4. Exit")
    choice = input("Choose (1-4): ").strip()
    if choice == "1":
        run_sprite_tracker()
    elif choice == "2":
        run_flock_tracker()
    elif choice == "3":
        run_sprite_tracker()
        run_flock_tracker()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()

