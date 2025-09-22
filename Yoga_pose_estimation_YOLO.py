import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import torch
from ultralytics import YOLO
import os
import pickle
import json
from datetime import datetime
import os
import argparse

# Define the COCO keypoint names and indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Create a mapping from keypoint indices to names
KEYPOINT_DICT = {i: name for i, name in enumerate(KEYPOINT_NAMES)}

# Define the COCO keypoint skeleton connections (pairs of keypoint indices that form lines)
# Note: COCO keypoints are 0-indexed in YOLOv11x output
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], 
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], 
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# Define angles to calculate (triplets of keypoints forming angles)
# Format: [joint_idx, central_idx, joint_idx, angle_name]
# The central_idx is the keypoint at which we measure the angle
ANGLES_TO_CALCULATE = [
    [7, 5, 11, "Left Shoulder Angle"],    # Left Shoulder Angle (elbow-shoulder-hip)
    [12, 6, 8, "Right Shoulder Angle"],   # Right Shoulder Angle (hip-shoulder-elbow)
    [5, 7, 9, "Left Elbow Angle"],        # Left Elbow Angle (shoulder-elbow-wrist)
    [6, 8, 10, "Right Elbow Angle"],      # Right Elbow Angle (shoulder-elbow-wrist)
    [11, 13, 15, "Left Knee Angle"],      # Left Knee Angle (hip-knee-ankle)
    [12, 14, 16, "Right Knee Angle"],     # Right Knee Angle (hip-knee-ankle)
    [5, 11, 13, "Left Hip Angle"],        # Left Hip Angle (shoulder-hip-knee)
    [6, 12, 14, "Right Hip Angle"],       # Right Hip Angle (shoulder-hip-knee)
]

# Colors for visualization (one distinct color per keypoint)
POSE_PALETTE = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], 
    [230, 230, 0], [255, 153, 255], [153, 204, 255], 
    [255, 102, 255], [255, 51, 255], [102, 178, 255], 
    [51, 153, 255], [255, 153, 153], [255, 102, 102], 
    [255, 51, 51], [153, 255, 153], [102, 255, 102], 
    [51, 255, 51], [0, 255, 0]
], dtype=np.uint8)

# Initialize an empty dictionary for reference angles
# This will be filled dynamically from expert demonstrations
ground_truth_angles = {}

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees).
    The angle is measured at point b.
    
    Args:
        a, b, c: Points as [x, y] coordinates
        
    Returns:
        Angle in degrees
    """
    # Create vectors from points
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    # Calculate the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Handle potential numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle

def calculate_pose_angles(keypoints, confidence_threshold=0.5):
    """
    Calculate a set of predefined angles for pose analysis.
    
    Args:
        keypoints: Numpy array of keypoints with shape [n, 17, 3] where each keypoint is [x, y, confidence]
        confidence_threshold: Minimum confidence to consider a keypoint valid
        
    Returns:
        Dictionary of angles for each person detected
    """
    results = []
    
    # Check if keypoints array is empty or has invalid shape
    if keypoints.size == 0 or len(keypoints.shape) != 3 or keypoints.shape[1] != 17 or keypoints.shape[2] != 3:
        print(f"Warning: Invalid keypoints shape: {keypoints.shape if keypoints.size > 0 else 'empty array'}")
        return results
    
    # For each person in the image
    for person_idx in range(keypoints.shape[0]):
        person_angles = {}
        kpts = keypoints[person_idx]
        
        # Verify that we have the expected number of keypoints
        if kpts.shape[0] != 17:
            print(f"Warning: Expected 17 keypoints but got {kpts.shape[0]} for person {person_idx}")
            continue
        
        # Calculate each predefined angle
        for a_idx, center_idx, c_idx, angle_name in ANGLES_TO_CALCULATE:
            # Verify indices are within bounds
            if a_idx >= len(kpts) or center_idx >= len(kpts) or c_idx >= len(kpts):
                print(f"Warning: Index out of bounds for angle {angle_name}")
                person_angles[angle_name] = None
                continue
                
            # Get coordinates for the three points
            x1, y1, conf1 = kpts[a_idx]
            x2, y2, conf2 = kpts[center_idx]
            x3, y3, conf3 = kpts[c_idx]
            
            # Only calculate angle if all keypoints have sufficient confidence
            if conf1 > confidence_threshold and conf2 > confidence_threshold and conf3 > confidence_threshold:
                angle = calculate_angle([x1, y1], [x2, y2], [x3, y3])
                person_angles[angle_name] = angle
            else:
                person_angles[angle_name] = None  # Mark as not calculable
        
        results.append(person_angles)
    
    return results

def compare_with_ground_truth(calculated_angles, ground_truth, tolerance=10.0):
    """
    Compare calculated angles with ground truth values within a tolerance.
    
    Args:
        calculated_angles: Dictionary of calculated angles
        ground_truth: Dictionary of ground truth angles
        tolerance: Acceptable difference in degrees
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for angle_name, calc_value in calculated_angles.items():
        if calc_value is None:
            results[angle_name] = {
                "calculated": None,
                "ground_truth": ground_truth.get(angle_name),
                "difference": None,
                "within_tolerance": False,
                "message": "Not calculable (missing keypoints)"
            }
            continue
            
        gt_value = ground_truth.get(angle_name)
        if gt_value is None:
            # No ground truth for this angle
            results[angle_name] = {
                "calculated": calc_value,
                "ground_truth": None,
                "difference": None,
                "within_tolerance": False,
                "message": "No ground truth available"
            }
            continue
            
        difference = abs(calc_value - gt_value)
        within_tolerance = difference <= tolerance
        
        results[angle_name] = {
            "calculated": calc_value,
            "ground_truth": gt_value,
            "difference": difference,
            "within_tolerance": within_tolerance,
            "message": f"{'Within' if within_tolerance else 'Outside'} tolerance (±{tolerance}°)"
        }
    
    return results

def generate_reference_angles(expert_dataset_path, output_directory=None, confidence_threshold=0.5, save_keypoints=True):
    """
    Generate reference angles from expert demonstrations for each yoga pose.
    
    Args:
        expert_dataset_path: Path to directory with expert pose demonstrations
        output_directory: Directory to save reference angles and keypoints (defaults to current directory)
        confidence_threshold: Minimum confidence to consider a keypoint valid
        save_keypoints: Whether to save the detected keypoints for reference poses
        
    Returns:
        Dictionary of reference angles for each pose
    """
    # Set default output directory if not specified
    if output_directory is None:
        output_directory = os.getcwd()  # Use current working directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
        
    reference_angles = {}
    angle_stds = {}  # To track standard deviations
    all_keypoints = {}  # To store keypoints for each pose
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Generating reference angles from expert demonstrations in {expert_dataset_path}")
    print(f"Results will be saved to {output_directory}")
    
    # Load the YOLOv11x pose model
    model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
    
    # Process each pose class directory
    for pose_dir in os.listdir(expert_dataset_path):
        pose_path = os.path.join(expert_dataset_path, pose_dir)
        
        # Skip if not a directory
        if not os.path.isdir(pose_path):
            continue
            
        pose_name = pose_dir.replace('_', ' ').title()
        print(f"\nProcessing pose: {pose_name}")
        
        # Initialize angle collectors and keypoints storage
        angle_data = {angle_name: [] for _, _, _, angle_name in ANGLES_TO_CALCULATE}
        pose_keypoints = []
        
        # Get all image files in this pose directory
        image_files = [f for f in os.listdir(pose_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"  No images found for {pose_name}")
            continue
            
        print(f"  Found {len(image_files)} demonstration images")
        
        # Process each expert demonstration
        for img_file in image_files:
            img_path = os.path.join(pose_path, img_file)
            
            try:
                # Process the image with YOLOv11x
                results = model(img_path)
                
                if len(results) == 0 or len(results[0].keypoints) == 0:
                    print(f"  No pose detected in: {img_file}")
                    continue
                
                # Extract keypoints from results
                keypoints = results[0].keypoints.data.cpu().numpy()  # Shape: [n, 17, 3] where each point is [x, y, confidence]
                
                if save_keypoints:
                    # Store keypoints with filename reference
                    pose_keypoints.append({
                        'filename': img_file,
                        'keypoints': keypoints.tolist()  # Convert to list for JSON serialization
                    })
                
                # Calculate pose angles
                pose_angles = calculate_pose_angles(keypoints, confidence_threshold)
                
                if pose_angles and len(pose_angles) > 0:
                    # Collect angles from this example
                    for angle_name, angle_value in pose_angles[0].items():
                        if angle_value is not None:
                            angle_data[angle_name].append(angle_value)
            
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
        
        # Calculate mean and standard deviation of angles for this pose
        pose_reference = {}
        pose_std = {}
        
        for angle_name, values in angle_data.items():
            if values:  # Only if we have collected values
                pose_reference[angle_name] = np.mean(values)
                pose_std[angle_name] = np.std(values)
                print(f"  {angle_name}: {pose_reference[angle_name]:.2f}° (±{pose_std[angle_name]:.2f}°) from {len(values)} samples")
            else:
                print(f"  {angle_name}: No valid measurements")
        
        reference_angles[pose_name] = pose_reference
        angle_stds[pose_name] = pose_std
        
        # Store keypoints for this pose
        if save_keypoints and pose_keypoints:
            all_keypoints[pose_name] = pose_keypoints
    
    print(f"\nReference angles generated for {len(reference_angles)} poses")
    
    # Create filenames with timestamp to avoid overwriting
    reference_pkl_filename = os.path.join(output_directory, f'reference_angles_{timestamp}.pkl')
    reference_json_filename = os.path.join(output_directory, f'reference_angles_{timestamp}.json')
    keypoints_filename = os.path.join(output_directory, f'reference_keypoints_{timestamp}.json')
    
    # Save as pickle (preserves all data types)
    with open(reference_pkl_filename, 'wb') as f:
        pickle.dump({'angles': reference_angles, 'stds': angle_stds}, f)
    print(f"Saved reference angles to {reference_pkl_filename}")
    
    # Also save as JSON for human readability (with some formatting)
    try:
        with open(reference_json_filename, 'w') as f:
            json.dump({'angles': reference_angles, 'stds': angle_stds}, f, indent=2)
        print(f"Saved human-readable reference angles to {reference_json_filename}")
    except Exception as e:
        print(f"Could not save JSON version: {e}")
    
    # Save keypoints if requested
    if save_keypoints and all_keypoints:
        try:
            with open(keypoints_filename, 'w') as f:
                json.dump(all_keypoints, f)
            print(f"Saved reference keypoints to {keypoints_filename}")
        except Exception as e:
            print(f"Could not save keypoints: {e}")
    
    return reference_angles

def load_reference_angles(path):
    """
    Load reference angles from a saved file.
    
    Args:
        path: Path to the file containing reference angles
        
    Returns:
        Dictionary of reference angles for each pose
    """
    # If path is a directory, look for the reference_angles.pkl file
    if os.path.isdir(path):
        path = os.path.join(path, 'reference_angles.pkl')
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Support both data structures
        if isinstance(data, dict) and 'angles' in data:
            print(f"Found 'angles' key in reference file with {len(data['angles'])} poses")
            return data['angles']
        else:
            # Handle the case where the data is directly a dictionary of poses
            if isinstance(data, dict):
                print(f"Loaded reference data with {len(data)} poses")
                return data
            else:
                print(f"Unexpected data format in reference file: {type(data)}")
                return {}
            
    except Exception as e:
        print(f"Error loading reference angles: {e}")
        return {}
    
def find_pose_in_references(pose_name, reference_angles):
    """
    Try to find a matching pose name in the reference angles.
    
    Args:
        pose_name: The pose name to look for
        reference_angles: Dictionary of reference angles
        
    Returns:
        Tuple of (matched_name, reference_angles_for_pose, found_match)
    """
    if not reference_angles:
        return pose_name, {}, False
        
    # Helper function for normalizing pose names
    def normalize_pose_name(name):
        return name.lower().replace('_', ' ').replace('-', ' ')
    
    # Convert raw classifier output to display format
    # Example: "tree_pose" -> "Tree Pose"
    words = pose_name.replace('_', ' ').replace('-', ' ').split()
    display_format = ' '.join(word.capitalize() for word in words)
    
    # 1. Check direct match with original name
    if pose_name in reference_angles:
        return pose_name, reference_angles[pose_name], True
        
    # 2. Check direct match with display format
    if display_format in reference_angles:
        return display_format, reference_angles[display_format], True
    
    # 3. Try case-insensitive and format-insensitive matching
    normalized_input = normalize_pose_name(pose_name)
    
    for ref_name in reference_angles.keys():
        normalized_ref = normalize_pose_name(ref_name)
        
        # Exact match after normalization
        if normalized_input == normalized_ref:
            return ref_name, reference_angles[ref_name], True
        
        # Check if one is a subset of the other (for hyphenated variations)
        if normalized_input in normalized_ref or normalized_ref in normalized_input:
            return ref_name, reference_angles[ref_name], True
    
    # No match found
    return display_format, {}, False

def plot_2d_pose_with_angles(ax, keypoints, angles_dict, comparison_results=None, confidence_threshold=0.5):
    """
    Plot 2D pose keypoints, connections, and joint angles on a matplotlib axis.
    Highlight angles that don't match ground truth.
    
    Args:
        ax: Matplotlib axis to plot on
        keypoints: Numpy array of keypoints with shape [n, 17, 3] where each keypoint is [x, y, confidence]
        angles_dict: Dictionary of calculated angles
        comparison_results: Dictionary with angle comparison results
        confidence_threshold: Minimum confidence to display a keypoint
    """
    # Create a blank white canvas
    ax.set_facecolor('white')
    
    # For each person in the image
    for person_idx in range(keypoints.shape[0]):
        # Get keypoints for this person
        kpts = keypoints[person_idx]
        person_angles = angles_dict[person_idx]
        
        # Draw connections between keypoints (the skeleton)
        for pair_idx, (idx1, idx2) in enumerate(SKELETON):
            # Get coordinates and confidence for both points
            x1, y1, conf1 = kpts[idx1]
            x2, y2, conf2 = kpts[idx2]
            
            # Skip if either point has low confidence
            if conf1 < confidence_threshold or conf2 < confidence_threshold:
                continue
            
            # Get color for this connection
            color = POSE_PALETTE[pair_idx % len(POSE_PALETTE)].tolist()
            # Convert color from 0-255 range to 0-1 range for matplotlib
            color = [c/255.0 for c in color]
            
            # Draw a line between the two points
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
        
        # Draw each keypoint
        for idx, (x, y, conf) in enumerate(kpts):
            # Skip low-confidence or non-existent keypoints
            if conf < confidence_threshold:
                continue
            
            # Get color for this keypoint
            color = POSE_PALETTE[idx % len(POSE_PALETTE)].tolist()
            # Convert color from 0-255 range to 0-1 range for matplotlib
            color = [c/255.0 for c in color]
            
            # Draw a circle at the keypoint location
            ax.scatter(x, y, s=50, color=color, marker='o')
            
            # Optionally add keypoint number for debugging
            ax.text(x+5, y+5, str(idx), fontsize=8, color=color)
        
        # Add angle annotations for key joints
        for angle_def in ANGLES_TO_CALCULATE:
            p1_idx, center_idx, p3_idx, angle_name = angle_def
            
            # Get keypoint positions
            x1, y1, conf1 = kpts[p1_idx]
            x2, y2, conf2 = kpts[center_idx]
            x3, y3, conf3 = kpts[p3_idx]
            
            # Check if we have a valid angle
            if angle_name in person_angles and person_angles[angle_name] is not None:
                angle_value = person_angles[angle_name]
                
                # Place the angle text in a position that's visible but not on top of points
                # Find midpoint between the two rays
                mid_x = (x1 + x3) / 2
                mid_y = (y1 + y3) / 2
                
                # Place text in direction towards the midpoint from the center
                dx = (mid_x - x2) * 0.5
                dy = (mid_y - y2) * 0.5
                
                # Determine if this angle is within tolerance (if comparison results available)
                within_tolerance = True  # Default to true if no comparison
                if comparison_results and angle_name in comparison_results:
                    within_tolerance = comparison_results[angle_name]["within_tolerance"]
                
                # Select text color based on whether angle is within tolerance
                text_color = 'black' if within_tolerance else 'red'
                text_bg_color = 'white' if within_tolerance else 'pink'
                
                # Add angle text with a background for better visibility
                ax.text(x2 + dx, y2 + dy, f'{angle_value:.1f}°', 
                        fontsize=9, color=text_color,
                        bbox=dict(facecolor=text_bg_color, alpha=0.7, edgecolor='none', pad=1))
                
                # Draw a red circle around incorrect angles
                if not within_tolerance and comparison_results:
                    # Create a red circle around the joint center
                    circle = plt.Circle((x2, y2), radius=20, color='red', fill=False, linewidth=2)
                    ax.add_patch(circle)
    
    # Invert y-axis to match image coordinates (origin at top-left)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('2D Pose with Angles')

def display_pose_comparison(comparison_results, tolerance):
    """
    Display a comparison of calculated pose angles vs ground truth.
    
    Args:
        comparison_results: Dictionary with comparison results
        tolerance: The tolerance used for comparison
    """
    print(f"\nPOSE COMPARISON (Tolerance: ±{tolerance}°)")
    print("=" * 60)
    
    correct_count = 0
    incorrect_count = 0
    unmeasurable_count = 0
    
    for angle_name, result in comparison_results.items():
        calc_value = result["calculated"]
        gt_value = result["ground_truth"]
        difference = result["difference"]
        within_tolerance = result["within_tolerance"]
        message = result["message"]
        
        status = "✓" if within_tolerance else "✗"
        
        if calc_value is None:
            unmeasurable_count += 1
        elif within_tolerance:
            correct_count += 1
        else:
            incorrect_count += 1
        
        if calc_value is not None and gt_value is not None:
            print(f"{status} {angle_name}:")
            print(f"  Calculated: {calc_value:.2f}°")
            print(f"  Ground Truth: {gt_value:.2f}°")
            print(f"  Difference: {difference:.2f}°")
            print(f"  Status: {message}")
        else:
            print(f"- {angle_name}: {message}")
        
        print()
    
    total_measurable = correct_count + incorrect_count
    if total_measurable > 0:
        accuracy = (correct_count / total_measurable) * 100
        print(f"SUMMARY: {correct_count}/{total_measurable} angles correct ({accuracy:.1f}%)")
        if unmeasurable_count > 0:
            print(f"Note: {unmeasurable_count} angles could not be measured due to missing keypoints.")

def provide_correction_feedback(comparison_results, tolerance):
    """
    Generate verbal feedback for pose correction based on comparison results.
    
    Args:
        comparison_results: Dictionary with angle comparison results
        tolerance: The tolerance used for comparison
        
    Returns:
        List of correction instructions
    """
    feedback = []
    
    for angle_name, result in comparison_results.items():
        calc_value = result["calculated"]
        gt_value = result["ground_truth"]
        within_tolerance = result["within_tolerance"]
        
        if calc_value is None or gt_value is None or within_tolerance:
            continue
        
        # Calculate the direction and magnitude of correction
        diff = gt_value - calc_value
        magnitude = abs(diff)
        direction = "increase" if diff > 0 else "decrease"
        
        # Determine the body part and action based on the angle name
        if "Shoulder" in angle_name:
            body_part = angle_name.split()[0] + " arm"
            action = f"{direction} your shoulder angle by raising your arm" if diff > 0 else f"{direction} your shoulder angle by lowering your arm"
        elif "Elbow" in angle_name:
            body_part = angle_name.split()[0] + " arm"
            action = f"{'straighten' if diff > 0 else 'bend'} your elbow more"
        elif "Knee" in angle_name:
            body_part = angle_name.split()[0] + " leg"
            action = f"{'straighten' if diff > 0 else 'bend'} your knee more"
        elif "Hip" in angle_name:
            body_part = angle_name.split()[0] + " hip"
            action = f"{'widen' if diff > 0 else 'narrow'} your stance"
        else:
            body_part = angle_name
            action = f"{direction} the angle"
        
        # Create specific feedback message
        message = f"Adjust your {body_part}: {action} (by {magnitude:.1f}°)"
        feedback.append(message)
    
    # Sort feedback by priority (largest deviation first)
    return feedback

def analyze_yoga_pose(image_path, pose_name=None, angle_tolerance=10.0):
    """
    Analyze a yoga pose image using YOLOv11x and compare with ground truth if available.
    
    Args:
        image_path: Path to the image file
        pose_name: Name of the yoga pose (for ground truth comparison)
        angle_tolerance: Acceptable difference in degrees
        
    Returns:
        Tuple of (keypoints, pose_angles, comparison_results)
    """
    # Load the YOLOv11x pose model
    model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
    
    # Process the image
    results = model(image_path)
    
    if len(results) == 0 or len(results[0].keypoints) == 0:
        print(f"No pose detected in {image_path}")
        return None, None, None
    
    # Extract keypoints from results
    keypoints = results[0].keypoints.data.cpu().numpy()  # Shape: [n, 17, 3] where each point is [x, y, confidence]
    
    # Calculate pose angles
    pose_angles = calculate_pose_angles(keypoints)
    
    # Compare with ground truth if available
    comparison_results = None
    if pose_name and pose_name in ground_truth_angles:
        comparison_results = compare_with_ground_truth(pose_angles[0], ground_truth_angles[pose_name], angle_tolerance)
        # Display detailed pose comparison
        display_pose_comparison(comparison_results, angle_tolerance)
        
        # Generate and display correction feedback
        feedback = provide_correction_feedback(comparison_results, angle_tolerance)
        if feedback:
            print("\nPOSE CORRECTION FEEDBACK:")
            print("=" * 60)
            for i, msg in enumerate(feedback, 1):
                print(f"{i}. {msg}")
        else:
            print("\nGreat job! Your pose is correct within the tolerance range.")
    
    return keypoints, pose_angles, comparison_results

def visualize_pose_analysis(image_path, keypoints, pose_angles, comparison_results=None, pose_name=None, angle_tolerance=10.0):
    """
    Visualize the pose analysis results.
    
    Args:
        image_path: Path to the image file
        keypoints: Numpy array of keypoints
        pose_angles: Dictionary of calculated angles
        comparison_results: Dictionary with angle comparison results
        pose_name: Name of the yoga pose
        angle_tolerance: Acceptable difference in degrees
    """
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots - 1 row, 2 columns
    fig = plt.figure(figsize=(15, 7))
    
    # Original image subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image')
    ax1.set_axis_off()
    
    # 2D pose visualization subplot
    ax2 = fig.add_subplot(1, 2, 2)
    plot_2d_pose_with_angles(ax2, keypoints, pose_angles, comparison_results)
    
    # Add a main title with pose information
    if comparison_results:
        correct_count = sum(1 for res in comparison_results.values() if res["within_tolerance"])
        total_count = sum(1 for res in comparison_results.values() if res["calculated"] is not None)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        plt.suptitle(f"{pose_name if pose_name else 'Unknown'} Pose - Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} angles correct, ±{angle_tolerance}°)", 
                     fontsize=16)
    else:
        plt.suptitle(f"{pose_name if pose_name else 'Unknown'} Pose - Analysis (No Ground Truth)", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Show the plot
    plt.show()
    
    # Return the figure for potential saving
    return fig

def real_time_pose_analysis(pose_name=None, angle_tolerance=10.0, mirror_mode=True, model_path=None, references_path=None):
    """
    Real-time yoga pose analysis using webcam.
    
    Args:
        pose_name: Name of the yoga pose (for ground truth comparison)
        angle_tolerance: Acceptable difference in degrees
        mirror_mode: If True, flip the webcam feed horizontally
        model_path: Path to the trained classifier model for pose recognition
        references_path: Path to the reference angles file
    """
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Load classifier model if provided
        classifier = None
        pose_classes = []
        angle_features = []
        
        if model_path and os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                classifier = model_data['model']
                pose_classes = model_data['classes']
                angle_features = model_data['features']
                
                print(f"Loaded classifier model from {model_path}")
                print(f"Available classifier poses: {', '.join(pose_classes)}")
            except Exception as e:
                print(f"Error loading classifier model: {e}")
                classifier = None
        
        # Load reference angles from separate file if specified
        global ground_truth_angles
        if references_path and os.path.exists(references_path):
            try:
                ground_truth_angles = load_reference_angles(references_path)
                print(f"Loaded reference angles for {len(ground_truth_angles)} poses from {references_path}")
                print(f"Available reference poses: {', '.join(ground_truth_angles.keys())}")
            except Exception as e:
                print(f"Error loading reference angles: {e}")
                ground_truth_angles = {}
        
        # Variables for pose classification
        detected_pose = None
        detected_class = None
        pose_confidence = 0
        
        # Variables for the current pose and ground truth
        current_pose_name = None
        can_compare = False
        gt_angles = {}
        
        # Set the fixed pose name if provided
        fixed_pose_mode = pose_name is not None
        if fixed_pose_mode:
            print(f"Fixed pose mode activated with pose: {pose_name}")
            
            # Try to find the reference angles for this pose
            matched_name, ref_angles, found_match = find_pose_in_references(pose_name, ground_truth_angles)
            
            if found_match:
                current_pose_name = matched_name
                can_compare = True
                gt_angles = ref_angles
                print(f"Found reference angles for: {current_pose_name}")
                if matched_name != pose_name:
                    print(f"Note: '{pose_name}' was matched to '{matched_name}' in reference angles")
            else:
                print(f"Warning: No reference angles found for '{pose_name}'")
                print(f"Available reference poses: {', '.join(ground_truth_angles.keys())}")
        
        # Print debug message showing the first reference angle for debugging
        if ground_truth_angles and len(ground_truth_angles) > 0:
            first_pose = next(iter(ground_truth_angles))
            first_angles = ground_truth_angles[first_pose]
            print(f"Sample reference angle data for '{first_pose}':")
            print(f"  Contains {len(first_angles)} angle measurements")
            if len(first_angles) > 0:
                first_angle_name = next(iter(first_angles))
                print(f"  Sample angle: {first_angle_name} = {first_angles[first_angle_name]}")
        
        # Load the YOLOv11x pose model
        model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
        
        # Create a window
        cv2.namedWindow('YOLOv11x Pose Analysis', cv2.WINDOW_NORMAL)
        
        # Frame counter for classification frequency
        frame_count = 0
        
        # Variable to track if a person is in the frame
        person_detected = False
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Increment frame counter
            frame_count += 1
            
            # Mirror the image if needed
            if mirror_mode:
                frame = cv2.flip(frame, 1)
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            try:
                # Process the frame with YOLOv11x with error handling
                results = model(frame, verbose=False)
                
                # Reset person detection status for this frame
                person_detected = False
                
                # Check if any pose was detected
                if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
                    # Person is detected
                    person_detected = True
                    
                    # Extract keypoints from results
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    # Verify keypoints shape is valid
                    if keypoints.shape[0] == 0 or keypoints.shape[1] != 17 or keypoints.shape[2] != 3:
                        raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")
                    
                    # Calculate pose angles
                    pose_angles = calculate_pose_angles(keypoints)
                    
                    # Only classify if we got valid pose angles
                    if len(pose_angles) > 0 and pose_angles[0]:
                        # Classify the pose if we have a classifier (every 3 frames) and not in fixed pose mode
                        if not fixed_pose_mode and classifier is not None and frame_count % 3 == 0:
                            try:
                                # Check if we have all the necessary angle features
                                if not angle_features:
                                    print("Error: No angle features defined")
                                    detected_pose = None
                                    detected_class = None
                                    pose_confidence = 0
                                else:
                                    # Count how many valid angles we have
                                    valid_angles = sum(1 for angle_name in angle_features
                                                     if angle_name in pose_angles[0] and pose_angles[0][angle_name] is not None)
                                    
                                    # Only classify if we have at least 4 valid angles
                                    if valid_angles >= 4:
                                        # Extract features for classification
                                        feature_vector = []
                                        for angle_name in angle_features:
                                            angle_value = pose_angles[0].get(angle_name)
                                            feature_vector.append(angle_value if angle_value is not None else 0)
                                        
                                        # Reshape for sklearn
                                        feature_vector = np.array(feature_vector).reshape(1, -1)
                                        
                                        # Get prediction and probability
                                        prediction = classifier.predict(feature_vector)[0]
                                        probabilities = classifier.predict_proba(feature_vector)[0]
                                        confidence = probabilities[prediction]
                                        
                                        # Only accept classifications with decent confidence
                                        if confidence >= 0.4:
                                            # Get the classifier output (e.g., "tree_pose")
                                            predicted_class = pose_classes[prediction]
                                            detected_class = predicted_class
                                            
                                            # Format for display (e.g., "Tree Pose")
                                            display_name = ' '.join(word.capitalize() for word in predicted_class.replace('_', ' ').replace('-', ' ').split())
                                            detected_pose = display_name
                                            pose_confidence = confidence
                                            
                                            # Try to find matching reference angles
                                            matched_name, ref_angles, found_match = find_pose_in_references(
                                                predicted_class, ground_truth_angles
                                            )
                                            
                                            if found_match:
                                                current_pose_name = matched_name
                                                can_compare = True
                                                gt_angles = ref_angles
                                            else:
                                                current_pose_name = display_name
                                                can_compare = False
                                                gt_angles = {}
                                                
                                            print(f"Frame {frame_count}: Detected class: '{predicted_class}' → '{display_name}' with confidence: {confidence*100:.1f}%")
                                            if found_match:
                                                print(f"  - Found matching reference angles with key: '{matched_name}'")
                                            else:
                                                print(f"  - No matching reference angles found")
                                        else:
                                            # Confidence too low, consider no pose detected
                                            detected_pose = None
                                            detected_class = None
                                            pose_confidence = 0
                                            current_pose_name = None
                                            can_compare = False
                                            gt_angles = {}
                                            print(f"Frame {frame_count}: Low confidence detection ({confidence*100:.1f}%) - ignoring")
                                    else:
                                        # Not enough valid angles, consider no pose detected
                                        detected_pose = None
                                        detected_class = None
                                        pose_confidence = 0
                                        current_pose_name = None
                                        can_compare = False
                                        gt_angles = {}
                                        print(f"Frame {frame_count}: Not enough valid angles ({valid_angles}) for classification")
                            except Exception as e:
                                print(f"Error during pose classification: {e}")
                                detected_pose = None
                                detected_class = None
                                pose_confidence = 0
                                current_pose_name = None
                                can_compare = False
                                gt_angles = {}
                        
                        # Draw connections between keypoints (the skeleton)
                        for person_idx in range(min(keypoints.shape[0], len(pose_angles))):
                            # Get keypoints for this person
                            kpts = keypoints[person_idx]
                            
                            # Remove confidence value indicator label
                            if person_idx == 0:
                                # Clear the top-left area where the "person 0.93" text appears
                                cv2.rectangle(display_frame, (0, 0), (150, 30), (0, 0, 0), -1)
                            
                            # Draw connections between keypoints (the skeleton)
                            for pair_idx, (idx1, idx2) in enumerate(SKELETON):
                                if idx1 >= len(kpts) or idx2 >= len(kpts):
                                    continue  # Skip if indices are out of bounds
                                
                                # Get coordinates and confidence for both points
                                x1, y1, conf1 = kpts[idx1]
                                x2, y2, conf2 = kpts[idx2]
                                
                                # Skip if either point has low confidence
                                if conf1 < 0.5 or conf2 < 0.5:
                                    continue
                                
                                # Get color for this connection - convert to BGR for OpenCV
                                color = POSE_PALETTE[pair_idx % len(POSE_PALETTE)].tolist()
                                # OpenCV uses BGR
                                color = [color[2], color[1], color[0]]
                                
                                # Draw a line between the two points
                                pt1 = (int(x1), int(y1))
                                pt2 = (int(x2), int(y2))
                                cv2.line(display_frame, pt1, pt2, color, 2)
                            
                            # Draw each keypoint
                            for idx, (x, y, conf) in enumerate(kpts):
                                # Skip low-confidence keypoints
                                if conf < 0.5:
                                    continue
                                
                                # Get color for this keypoint - convert to BGR for OpenCV
                                color = POSE_PALETTE[idx % len(POSE_PALETTE)].tolist()
                                # OpenCV uses BGR
                                color = [color[2], color[1], color[0]]
                                
                                # Draw a circle at the keypoint location
                                cv2.circle(display_frame, (int(x), int(y)), 5, color, -1)
                            
                            # Add angle annotations for key joints
                            for angle_def in ANGLES_TO_CALCULATE:
                                if len(angle_def) < 4:
                                    continue  # Skip invalid angle definitions
                                
                                p1_idx, center_idx, p3_idx, angle_name = angle_def
                                
                                if (p1_idx >= len(kpts) or center_idx >= len(kpts) or p3_idx >= len(kpts)):
                                    continue  # Skip if indices are out of bounds
                                
                                # Get keypoint positions
                                x1, y1, conf1 = kpts[p1_idx]
                                x2, y2, conf2 = kpts[center_idx]
                                x3, y3, conf3 = kpts[p3_idx]
                                
                                # Only calculate angle if all keypoints have sufficient confidence
                                if conf1 > 0.5 and conf2 > 0.5 and conf3 > 0.5:
                                    # Check if this angle exists in the calculated angles
                                    if (person_idx < len(pose_angles) and 
                                        angle_name in pose_angles[person_idx] and 
                                        pose_angles[person_idx][angle_name] is not None):
                                        
                                        angle_value = pose_angles[person_idx][angle_name]
                                        
                                        # Determine if this angle is within tolerance (if comparison results available)
                                        within_tolerance = True
                                        if can_compare and gt_angles and angle_name in gt_angles:
                                            gt_value = gt_angles.get(angle_name)
                                            if gt_value is not None:
                                                within_tolerance = abs(angle_value - gt_value) <= angle_tolerance
                                        
                                        # Select text color based on whether angle is within tolerance
                                        text_color = (0, 255, 0) if within_tolerance else (0, 0, 255)
                                        
                                        # Place text near the joint
                                        text_pos = (int(x2), int(y2 - 10))
                                        cv2.putText(display_frame, f'{angle_value:.1f}°', text_pos, 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                                        
                                        # Draw a circle around incorrect angles
                                        if not within_tolerance and can_compare and angle_name in gt_angles:
                                            cv2.circle(display_frame, (int(x2), int(y2)), 20, (0, 0, 255), 2)
                        
                        # Compare with ground truth if available
                        comparison_results = None
                        if can_compare and gt_angles and len(pose_angles) > 0:
                            comparison_results = compare_with_ground_truth(pose_angles[0], gt_angles, angle_tolerance)
                            
                            # Generate correction feedback
                            feedback = provide_correction_feedback(comparison_results, angle_tolerance)
                            
                            # Display feedback on the image
                            y_pos = 30
                            for i, msg in enumerate(feedback[:3]):  # Show only top 3 corrections
                                cv2.putText(display_frame, msg, (10, y_pos), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                y_pos += 30
                            
                            # Calculate accuracy
                            correct_count = sum(1 for res in comparison_results.values() 
                                              if res["within_tolerance"] and res["ground_truth"] is not None)
                            total_count = sum(1 for res in comparison_results.values() 
                                            if res["calculated"] is not None and res["ground_truth"] is not None)
                            
                            # Only display accuracy if we have valid comparisons
                            if total_count > 0:
                                accuracy = (correct_count / total_count * 100)
                                accuracy_text = f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} angles correct, ±{angle_tolerance}°)"
                                accuracy_color = (0, 255, 0) if accuracy >= 75 else (0, 165, 255) if accuracy >= 50 else (0, 0, 255)
                            else:
                                accuracy_text = f"Accuracy: N/A (No matching angle data)"
                                accuracy_color = (0, 165, 255)
                        else:
                            if fixed_pose_mode:
                                accuracy_text = f"No reference angles found for '{pose_name}'"
                            else:
                                if detected_pose:
                                    if detected_class:
                                        accuracy_text = f"No reference angles found for '{detected_class}'"
                                    else:
                                        accuracy_text = f"No reference angles found for '{detected_pose}'"
                                else:
                                    accuracy_text = "No reference angles available for comparison"
                            accuracy_color = (0, 165, 255)
                        
                        # Display accuracy information
                        cv2.putText(display_frame, accuracy_text, 
                                  (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, accuracy_color, 2)
                        
                        # Display pose name
                        if fixed_pose_mode:
                            pose_text = f"Fixed Pose: {pose_name}"
                            cv2.putText(display_frame, pose_text, 
                                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        elif detected_pose and pose_confidence > 0:
                            pose_text = f"Pose: {detected_pose} (Confidence: {pose_confidence*100:.1f}%)"
                            cv2.putText(display_frame, pose_text, 
                                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            # No pose detected but person is in frame
                            cv2.putText(display_frame, "No pose detected", 
                                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # No person detected - Add message to inform user
                    person_detected = False
                    detected_pose = None
                    detected_class = None
                    pose_confidence = 0
                    current_pose_name = None
                    can_compare = False
                    gt_angles = {}
                    cv2.putText(display_frame, "No person detected", (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            except Exception as e:
                # Catch any errors during frame processing
                print(f"Error processing frame: {e}")
                # Add error message to the frame
                cv2.putText(display_frame, f"Error: {str(e)[:50]}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Reset state variables to avoid persistent errors
                detected_pose = None
                detected_class = None
                pose_confidence = 0
                current_pose_name = None
                can_compare = False
                gt_angles = {}
            
            # Display the frame
            cv2.imshow('YOLOv11x Pose Analysis', display_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Critical error in real_time_pose_analysis: {e}")
        # Try to clean up resources
        try:
            if 'cap' in locals() and cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
        except:
            pass

def normalize_keypoints(keypoints, confidence_threshold=0.5):
    """
    Normalize keypoints to make them invariant to scale and position.
    
    Args:
        keypoints: Numpy array of keypoints with shape [n, 17, 3] where each keypoint is [x, y, confidence]
        confidence_threshold: Minimum confidence to consider a keypoint valid
        
    Returns:
        Normalized keypoints with the same shape
    """
    normalized_keypoints = np.copy(keypoints)
    
    # For each person in the image
    for person_idx in range(keypoints.shape[0]):
        kpts = keypoints[person_idx]
        
        # Find the center of the torso (midpoint between shoulders and hips)
        left_shoulder = kpts[5]  # left_shoulder
        right_shoulder = kpts[6]  # right_shoulder
        left_hip = kpts[11]  # left_hip
        right_hip = kpts[12]  # right_hip
        
        # Check if all necessary keypoints are available
        if (left_shoulder[2] < confidence_threshold or right_shoulder[2] < confidence_threshold or
            left_hip[2] < confidence_threshold or right_hip[2] < confidence_threshold):
            continue  # Skip normalization if critical points are missing
        
        # Calculate reference center point (center of torso)
        center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
        
        # Calculate scaling factor based on torso size
        # Distance between shoulders
        shoulder_dist = np.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                               (left_shoulder[1] - right_shoulder[1])**2)
        # Distance between hips
        hip_dist = np.sqrt((left_hip[0] - right_hip[0])**2 + 
                          (left_hip[1] - right_hip[1])**2)
        # Average torso width
        torso_width = (shoulder_dist + hip_dist) / 2
        
        # Normalize all keypoints relative to center and scale
        for kpt_idx in range(kpts.shape[0]):
            # Only normalize if the keypoint is valid
            if kpts[kpt_idx, 2] >= confidence_threshold:
                # Translate to origin
                normalized_x = kpts[kpt_idx, 0] - center_x
                normalized_y = kpts[kpt_idx, 1] - center_y
                
                # Scale by torso width
                normalized_x = normalized_x / torso_width
                normalized_y = normalized_y / torso_width
                
                # Store normalized coordinates
                normalized_keypoints[person_idx, kpt_idx, 0] = normalized_x
                normalized_keypoints[person_idx, kpt_idx, 1] = normalized_y
    
    return normalized_keypoints

def train_pose_classifier(dataset_path, model_output_path, reference_angles_path=None, num_classes=8):
    """
    Train a simple Random Forest classifier on extracted pose angles.
    
    Args:
        dataset_path: Path to the directory containing pose images sorted by pose class
        model_output_path: Path to save the trained model
        reference_angles_path: Path to load pre-generated reference angles (optional)
        num_classes: Number of yoga pose classes
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import os
    import pickle
    import tempfile
    
    # Define the features to extract (all angles)
    angle_features = [
        "Left Shoulder Angle", "Right Shoulder Angle",
        "Left Elbow Angle", "Right Elbow Angle",
        "Left Knee Angle", "Right Knee Angle",
        "Left Hip Angle", "Right Hip Angle"
    ]
    
    # Initialize lists to store features and labels
    X = []  # Features (angles)
    y = []  # Labels (pose names)
    
    print("Processing dataset images...")
    
    # Load the YOLOv11x pose model
    model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
    
    # Get all pose class folders
    pose_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_idx, pose_class in enumerate(pose_classes):
        class_dir = os.path.join(dataset_path, pose_class)
        pose_name = pose_class.replace('_', ' ').title()
        print(f"Processing class {class_idx+1}/{len(pose_classes)}: {pose_name}")
        
        # Get all image files in this class
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Process image with YOLOv11x
                results = model(img_path, verbose=False)
                
                if len(results) > 0 and len(results[0].keypoints) > 0:
                    # Extract keypoints from results
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    # Calculate pose angles
                    pose_angles = calculate_pose_angles(keypoints)
                    
                    if len(pose_angles) > 0:
                        # Extract angle features
                        feature_vector = []
                        for angle_name in angle_features:
                            angle_value = pose_angles[0].get(angle_name)
                            # Use 0 as default if angle is not available
                            feature_vector.append(angle_value if angle_value is not None else 0)
                        
                        # Add to dataset
                        X.append(feature_vector)
                        y.append(class_idx)  # Use class index as label
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Check if we have enough data
    if len(X) < 10:
        print("Warning: Very small dataset. Classification may not be reliable.")
    
    # Load reference angles if provided, otherwise keep empty
    reference_angles = {}
    if reference_angles_path:
        print(f"Loading reference angles from {reference_angles_path}")
        try:
            reference_angles = load_reference_angles(reference_angles_path)
            print(f"Loaded reference angles for {len(reference_angles)} poses")
        except Exception as e:
            print(f"Error loading reference angles: {e}")
            print("Will continue without reference angles")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=pose_classes))
    
    # Get feature importance
    feature_importances = clf.feature_importances_
    print("\nFeature Importance:")
    for feature, importance in zip(angle_features, feature_importances):
        print(f"{feature}: {importance:.4f}")
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(os.path.abspath(model_output_path))
    os.makedirs(model_dir, exist_ok=True)
    
    # First write to a temporary file to avoid issues with read-only directories
    temp_model_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_model_file = temp_file.name
            # Save the model and class names along with reference angles
            pickle.dump({
                'model': clf, 
                'classes': pose_classes, 
                'features': angle_features,
                'reference_angles': reference_angles
            }, temp_file)
        
        # Then copy the temporary file to the final destination
        import shutil
        shutil.copy2(temp_model_file, model_output_path)
        print(f"Successfully saved model to {model_output_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
    finally:
        # Clean up the temporary file
        if temp_model_file and os.path.exists(temp_model_file):
            os.unlink(temp_model_file)
    
    return clf, pose_classes

def classify_pose(image_path, model_path):
    """
    Classify a yoga pose using the trained model.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model file
        
    Returns:
        Predicted pose class and confidence
    """
    import pickle
    import os
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return None, 0
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        clf = model_data['model']
        pose_classes = model_data['classes']
        angle_features = model_data['features']
        
        # Load reference angles from model file if available
        global ground_truth_angles
        if 'reference_angles' in model_data and model_data['reference_angles']:
            ground_truth_angles = model_data['reference_angles']
            print(f"Loaded reference angles for {len(ground_truth_angles)} poses from model file")
        
        # Load the YOLOv11x pose model
        yolo_model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
        
        # Process the image
        results = yolo_model(image_path, verbose=False)
        
        if len(results) == 0 or len(results[0].keypoints) == 0:
            print("No pose detected in the image.")
            return None, 0
        
        # Extract keypoints from results
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        # Calculate pose angles
        pose_angles = calculate_pose_angles(keypoints)
        
        if len(pose_angles) == 0:
            print("No valid pose angles could be calculated.")
            return None, 0
        
        # Extract features
        feature_vector = []
        for angle_name in angle_features:
            angle_value = pose_angles[0].get(angle_name)
            feature_vector.append(angle_value if angle_value is not None else 0)
        
        # Reshape for sklearn
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Get prediction and probability
        prediction = clf.predict(feature_vector)[0]
        probabilities = clf.predict_proba(feature_vector)[0]
        confidence = probabilities[prediction]
        
        # Convert class index to class name
        predicted_class = pose_classes[prediction]
        predicted_class_name = predicted_class.replace('_', ' ').title()
        
        print(f"Predicted pose: {predicted_class_name} (Confidence: {confidence*100:.2f}%)")
        
        # Print top 3 predictions if available
        if len(pose_classes) > 1:
            print("\nTop predictions:")
            top_indices = np.argsort(probabilities)[::-1][:min(3, len(pose_classes))]
            for i, idx in enumerate(top_indices):
                class_name = pose_classes[idx].replace('_', ' ').title()
                prob = probabilities[idx]
                print(f"{i+1}. {class_name}: {prob*100:.2f}%")
        
        return predicted_class_name, confidence
        
    except Exception as e:
        print(f"Error classifying pose: {e}")
        return None, 0

def process_video(video_path, pose_name=None, angle_tolerance=10.0, output_path=None, model_path=None):
    """
    Process a video file for yoga pose analysis with full angle calculation and feedback.
    
    Args:
        video_path: Path to the video file
        pose_name: Name of the yoga pose (for ground truth comparison)
        angle_tolerance: Acceptable difference in degrees
        output_path: Path to save the processed video
        model_path: Path to trained classifier model for pose recognition
    """
    import time
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output_path is provided
    writer = None
    if output_path:
        # Ensure proper file extension
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Warning: Could not initialize video writer for {output_path}")
            writer = None
    
    # Load classifier model if provided
    classifier = None
    pose_classes = []
    angle_features = []
    
    if model_path and os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            classifier = model_data['model']
            pose_classes = model_data['classes']
            angle_features = model_data['features']
            
            print(f"Loaded classifier model from {model_path}")
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            classifier = None
    
    # Variables for pose classification
    detected_pose = None
    detected_class = None
    pose_confidence = 0
    
    # Variables for the current pose and ground truth
    current_pose_name = None
    can_compare = False
    gt_angles = {}
    
    # Set the fixed pose name if provided
    fixed_pose_mode = pose_name is not None
    if fixed_pose_mode:
        print(f"Fixed pose mode activated with pose: {pose_name}")
        
        # Try to find the reference angles for this pose
        matched_name, ref_angles, found_match = find_pose_in_references(pose_name, ground_truth_angles)
        
        if found_match:
            current_pose_name = matched_name
            can_compare = True
            gt_angles = ref_angles
            print(f"Found reference angles for: {current_pose_name}")
        else:
            print(f"Warning: No reference angles found for '{pose_name}'")
    
    # Load the YOLOv11x pose model
    model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
    
    frame_count = 0
    processed_frames = 0
    
    # Frame rate calculation variables
    start_time = time.time()
    frame_times = []
    fps_display = 0.0
    fps_update_interval = 30  # Update FPS display every 30 frames
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {fps:.2f}")
    print(f"Output: {output_path if output_path else 'Display only'}")
    
    while cap.isOpened():
        frame_start_time = time.time()  # Start timing this frame
        
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        try:
            # Process the frame with YOLOv11x
            results = model(frame, verbose=False)
            
            # Reset person detection status for this frame
            person_detected = False
            
            # Check if any pose was detected
            if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
                # Person is detected
                person_detected = True
                
                # Extract keypoints from results
                keypoints = results[0].keypoints.data.cpu().numpy()
                
                # Verify keypoints shape is valid
                if keypoints.shape[0] == 0 or keypoints.shape[1] != 17 or keypoints.shape[2] != 3:
                    raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")
                
                # Calculate pose angles
                pose_angles = calculate_pose_angles(keypoints)
                
                # Only process if we got valid pose angles
                if len(pose_angles) > 0 and pose_angles[0]:
                    # Classify the pose if we have a classifier (every 5 frames) and not in fixed pose mode
                    if not fixed_pose_mode and classifier is not None and frame_count % 5 == 0:
                        try:
                            # Count how many valid angles we have
                            valid_angles = sum(1 for angle_name in angle_features
                                             if angle_name in pose_angles[0] and pose_angles[0][angle_name] is not None)
                            
                            # Only classify if we have at least 4 valid angles
                            if valid_angles >= 4:
                                # Extract features for classification
                                feature_vector = []
                                for angle_name in angle_features:
                                    angle_value = pose_angles[0].get(angle_name)
                                    feature_vector.append(angle_value if angle_value is not None else 0)
                                
                                # Reshape for sklearn
                                feature_vector = np.array(feature_vector).reshape(1, -1)
                                
                                # Get prediction and probability
                                prediction = classifier.predict(feature_vector)[0]
                                probabilities = classifier.predict_proba(feature_vector)[0]
                                confidence = probabilities[prediction]
                                
                                # Only accept classifications with decent confidence
                                if confidence >= 0.4:
                                    # Get the classifier output
                                    predicted_class = pose_classes[prediction]
                                    detected_class = predicted_class
                                    
                                    # Format for display
                                    display_name = ' '.join(word.capitalize() for word in predicted_class.replace('_', ' ').replace('-', ' ').split())
                                    detected_pose = display_name
                                    pose_confidence = confidence
                                    
                                    # Try to find matching reference angles
                                    matched_name, ref_angles, found_match = find_pose_in_references(
                                        predicted_class, ground_truth_angles
                                    )
                                    
                                    if found_match:
                                        current_pose_name = matched_name
                                        can_compare = True
                                        gt_angles = ref_angles
                                    else:
                                        current_pose_name = display_name
                                        can_compare = False
                                        gt_angles = {}
                        except Exception as e:
                            print(f"Error during pose classification in frame {frame_count}: {e}")
                    
                    # Draw connections between keypoints (the skeleton) - same as in webcam mode
                    for person_idx in range(min(keypoints.shape[0], len(pose_angles))):
                        # Get keypoints for this person
                        kpts = keypoints[person_idx]
                        
                        # Draw connections between keypoints (the skeleton)
                        for pair_idx, (idx1, idx2) in enumerate(SKELETON):
                            if idx1 >= len(kpts) or idx2 >= len(kpts):
                                continue
                            
                            # Get coordinates and confidence for both points
                            x1, y1, conf1 = kpts[idx1]
                            x2, y2, conf2 = kpts[idx2]
                            
                            # Skip if either point has low confidence
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                            
                            # Get color for this connection - convert to BGR for OpenCV
                            color = POSE_PALETTE[pair_idx % len(POSE_PALETTE)].tolist()
                            color = [color[2], color[1], color[0]]  # RGB to BGR
                            
                            # Draw a line between the two points
                            pt1 = (int(x1), int(y1))
                            pt2 = (int(x2), int(y2))
                            cv2.line(display_frame, pt1, pt2, color, 3)
                        
                        # Draw each keypoint
                        for idx, (x, y, conf) in enumerate(kpts):
                            if conf < 0.5:
                                continue
                            
                            # Get color for this keypoint - convert to BGR for OpenCV
                            color = POSE_PALETTE[idx % len(POSE_PALETTE)].tolist()
                            color = [color[2], color[1], color[0]]  # RGB to BGR
                            
                            # Draw a circle at the keypoint location
                            cv2.circle(display_frame, (int(x), int(y)), 6, color, -1)
                        
                        # Add angle annotations for key joints - same as in webcam mode
                        for angle_def in ANGLES_TO_CALCULATE:
                            if len(angle_def) < 4:
                                continue
                            
                            p1_idx, center_idx, p3_idx, angle_name = angle_def
                            
                            if (p1_idx >= len(kpts) or center_idx >= len(kpts) or p3_idx >= len(kpts)):
                                continue
                            
                            # Get keypoint positions
                            x1, y1, conf1 = kpts[p1_idx]
                            x2, y2, conf2 = kpts[center_idx]
                            x3, y3, conf3 = kpts[p3_idx]
                            
                            # Only calculate angle if all keypoints have sufficient confidence
                            if conf1 > 0.5 and conf2 > 0.5 and conf3 > 0.5:
                                # Check if this angle exists in the calculated angles
                                if (person_idx < len(pose_angles) and 
                                    angle_name in pose_angles[person_idx] and 
                                    pose_angles[person_idx][angle_name] is not None):
                                    
                                    angle_value = pose_angles[person_idx][angle_name]
                                    
                                    # Determine if this angle is within tolerance
                                    within_tolerance = True
                                    if can_compare and gt_angles and angle_name in gt_angles:
                                        gt_value = gt_angles.get(angle_name)
                                        if gt_value is not None:
                                            within_tolerance = abs(angle_value - gt_value) <= angle_tolerance
                                    
                                    # Select text color based on whether angle is within tolerance
                                    text_color = (0, 255, 0) if within_tolerance else (0, 0, 255)
                                    
                                    # Place text near the joint
                                    text_pos = (int(x2), int(y2 - 15))
                                    cv2.putText(display_frame, f'{angle_value:.1f}°', text_pos, 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                                    
                                    # Draw a circle around incorrect angles
                                    if not within_tolerance and can_compare and angle_name in gt_angles:
                                        cv2.circle(display_frame, (int(x2), int(y2)), 25, (0, 0, 255), 3)
                    
                    # Compare with ground truth if available
                    comparison_results = None
                    if can_compare and gt_angles and len(pose_angles) > 0:
                        comparison_results = compare_with_ground_truth(pose_angles[0], gt_angles, angle_tolerance)
                        
                        # Generate correction feedback
                        feedback = provide_correction_feedback(comparison_results, angle_tolerance)
                        
                        # Display feedback on the image
                        y_pos = 35
                        for i, msg in enumerate(feedback[:3]):  # Show only top 3 corrections
                            # Add background rectangle for better text visibility
                            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(display_frame, (5, y_pos - 25), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
                            cv2.putText(display_frame, msg, (10, y_pos), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            y_pos += 35
                        
                        # Calculate accuracy
                        correct_count = sum(1 for res in comparison_results.values() 
                                          if res["within_tolerance"] and res["ground_truth"] is not None)
                        total_count = sum(1 for res in comparison_results.values() 
                                        if res["calculated"] is not None and res["ground_truth"] is not None)
                        
                        # Only display accuracy if we have valid comparisons
                        if total_count > 0:
                            accuracy = (correct_count / total_count * 100)
                            accuracy_text = f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})"
                            accuracy_color = (0, 255, 0) if accuracy >= 75 else (0, 165, 255) if accuracy >= 50 else (0, 0, 255)
                        else:
                            accuracy_text = f"Accuracy: N/A (No matching angles)"
                            accuracy_color = (0, 165, 255)
                    else:
                        if fixed_pose_mode:
                            accuracy_text = f"No reference angles for '{pose_name}'"
                        else:
                            if detected_pose:
                                accuracy_text = f"No reference angles for '{detected_pose}'"
                            else:
                                accuracy_text = "No reference angles available"
                        accuracy_color = (0, 165, 255)
                    
                    # Display accuracy information with background
                    acc_text_size = cv2.getTextSize(accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (5, height - 50), (acc_text_size[0] + 15, height - 15), (0, 0, 0), -1)
                    cv2.putText(display_frame, accuracy_text, 
                              (10, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accuracy_color, 2)
                    
                    # Display pose name with background
                    if fixed_pose_mode:
                        pose_text = f"Fixed Pose: {pose_name}"
                    elif detected_pose and pose_confidence > 0:
                        pose_text = f"Pose: {detected_pose} ({pose_confidence*100:.1f}%)"
                    else:
                        pose_text = "Pose: Analyzing..."
                    
                    pose_text_size = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (5, height - 85), (pose_text_size[0] + 15, height - 55), (0, 0, 0), -1)
                    cv2.putText(display_frame, pose_text, 
                            (10, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    processed_frames += 1
                else:
                    # No valid pose angles calculated
                    cv2.putText(display_frame, "Pose detected but angles not calculable", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # No person detected
                cv2.putText(display_frame, "No person detected", 
                          (int(width/2) - 100, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Reset pose detection variables
                detected_pose = None
                detected_class = None
                pose_confidence = 0
                current_pose_name = None
                can_compare = False
                gt_angles = {}
        
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Add error message to frame
            cv2.putText(display_frame, f"Processing Error: {str(e)[:30]}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add frame counter and FPS for debugging
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                  (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(display_frame, f"Processing FPS: {fps_display:.1f}", 
                  (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Calculate frame processing time
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        frame_times.append(frame_processing_time)
        
        # Update FPS display every few frames
        if frame_count % fps_update_interval == 0 and len(frame_times) > 0:
            # Calculate average FPS over recent frames
            recent_times = frame_times[-fps_update_interval:] if len(frame_times) >= fps_update_interval else frame_times
            avg_frame_time = sum(recent_times) / len(recent_times)
            fps_display = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Print detailed frame rate info
            elapsed_time = time.time() - start_time
            overall_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Frame {frame_count}: Processing FPS: {fps_display:.1f}, Overall FPS: {overall_fps:.1f}, Frame time: {frame_processing_time*1000:.1f}ms")
        
        # Write the frame to output video if writer is initialized
        if writer and writer.isOpened():
            writer.write(display_frame)
        
        # Display progress with detailed timing info
        if frame_count % 30 == 0:  # Update every 30 frames
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            overall_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (total_frames - frame_count) / overall_fps if overall_fps > 0 else 0
            print(f"Progress: {progress:.1f}% | Processed poses: {processed_frames} | Overall FPS: {overall_fps:.1f} | ETA: {estimated_remaining:.1f}s")
    
    # Clean up
    cap.release()
    if writer and writer.isOpened():
        writer.release()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    overall_fps = frame_count / total_time if total_time > 0 else 0
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    
    print(f"\n{'='*60}")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with valid poses: {processed_frames}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Overall processing FPS: {overall_fps:.2f}")
    print(f"Average frame time: {avg_frame_time*1000:.1f} ms")
    print(f"Original video FPS: {fps:.2f}")
    print(f"Processing speed ratio: {overall_fps/fps:.2f}x {'(Real-time)' if overall_fps >= fps else '(Slower than real-time)'}")
    if output_path:
        print(f"Output saved to: {output_path}")
    print(f"{'='*60}")

# Additional function to benchmark processing performance
def benchmark_pose_processing(video_path, num_frames=100):
    """
    Benchmark pose processing performance on a subset of frames.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to process for benchmarking
    """
    import time
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Load the YOLOv11x pose model
    model = YOLO("/kaggle/input/yolov11-models/yolo11x-pose.pt")
    
    frame_times = []
    pose_detection_times = []
    angle_calculation_times = []
    
    print(f"Benchmarking pose processing on {num_frames} frames...")
    
    for i in range(num_frames):
        success, frame = cap.read()
        if not success:
            break
        
        frame_start = time.time()
        
        # Pose detection timing
        detection_start = time.time()
        results = model(frame, verbose=False)
        detection_time = time.time() - detection_start
        pose_detection_times.append(detection_time)
        
        # Angle calculation timing
        if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.data.cpu().numpy()
            if keypoints.shape[0] > 0 and keypoints.shape[1] == 17 and keypoints.shape[2] == 3:
                angle_start = time.time()
                pose_angles = calculate_pose_angles(keypoints)
                angle_time = time.time() - angle_start
                angle_calculation_times.append(angle_time)
        
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Calculate statistics
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        max_frame_time = max(frame_times)
        min_frame_time = min(frame_times)
        fps = 1.0 / avg_frame_time
        
        print(f"\n{'='*50}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Frames processed: {len(frame_times)}")
        print(f"Average frame time: {avg_frame_time*1000:.2f} ms")
        print(f"Min frame time: {min_frame_time*1000:.2f} ms")
        print(f"Max frame time: {max_frame_time*1000:.2f} ms")
        print(f"Processing FPS: {fps:.2f}")
        
        if pose_detection_times:
            avg_detection = sum(pose_detection_times) / len(pose_detection_times)
            print(f"Average pose detection time: {avg_detection*1000:.2f} ms")
        
        if angle_calculation_times:
            avg_angle = sum(angle_calculation_times) / len(angle_calculation_times)
            print(f"Average angle calculation time: {avg_angle*1000:.2f} ms")
        
        print(f"{'='*50}")
    else:
        print("No frames were processed successfully.")

def main():
    """
    Main function to demonstrate the yoga pose analysis system.
    """

    
    mode = 'webcam'
    # input_path = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\videos\1_Warrior 2_.mp4"
    # output_path = "new_video.mp4"
    pose_name = None
    # dataset_path = r"/kaggle/input/yoga-dataset/yoga_dataset"
    # expert_dataset = r"/kaggle/input/yoga-dataset/yoga_dataset"
    model_path = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\pose correction.pkl"
    references_path = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\angles_final.pkl"
    save_keypoints = True
    angle_tolerance = 10.0
    
    # Check for YOLOv11x pose model
    if not os.path.exists("yolo11x-pose.pt"):
        print("Downloading YOLOv11x pose model...")
        try:
            # Use subprocess to run curl command for download
            import subprocess
            subprocess.call(["curl", "-L", 
                           "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x-pose.pt", 
                           "-o", "yolo11x-pose.pt"])
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download the YOLOv11x pose model manually from: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x-pose.pt")
            return
    
    # Load reference angles if specified
    global ground_truth_angles
    if references_path and os.path.exists(references_path):
        print(f"Loading reference angles from {references_path}")
        ground_truth_angles = load_reference_angles(references_path)
    
    if mode == 'generate-references' and expert_dataset:
        # Generate reference angles from expert demonstrations
        output_dir = output_path if output_path else 'reference_data'
        os.makedirs(output_dir, exist_ok=True)
        ground_truth_angles = generate_reference_angles(
            expert_dataset, 
            output_directory=output_dir,
            save_keypoints=save_keypoints
        )
    
    elif mode == 'image' and input_path:
        # Process a single image
        keypoints, pose_angles, comparison_results = analyze_yoga_pose(input_path, pose_name)
        if keypoints is not None:
            fig = visualize_pose_analysis(input_path, keypoints, pose_angles, comparison_results, pose_name)
            if output_path:
                # Create output directory if needed
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # If model_path exists, also classify the pose
        if os.path.exists(model_path):
            predicted_pose, confidence = classify_pose(input_path, model_path)
            
            # If we have reference angles for the predicted pose, analyze it further
            if predicted_pose in ground_truth_angles:
                print(f"Analyzing pose accuracy against reference for {predicted_pose}...")
                keypoints, pose_angles, comparison_results = analyze_yoga_pose(
                    input_path, predicted_pose)
                
                if keypoints is not None:
                    fig = visualize_pose_analysis(
                        input_path, keypoints, pose_angles, comparison_results, 
                        predicted_pose, angle_tolerance=10.0)
                    
                    # Save the visualization if output path specified
                    if output_path:
                        # Create output directory if needed
                        output_dir = os.path.dirname(output_path)
                        if output_dir and not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    elif mode == 'video' and input_path:
        # Process a video file with enhanced functionality
        if output_path:
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        process_video(input_path, pose_name, 10, output_path, model_path)
    
    elif mode == 'webcam':
        # Real-time analysis using webcam
        # real_time_pose_analysis(pose_name, angle_tolerance=15.0, model_path=model_path, references_path=references_path)
        real_time_pose_analysis(pose_name=None, angle_tolerance=10.0, mirror_mode=True, model_path=model_path, references_path=references_path)
    
    elif mode == 'train' and dataset_path:
        # Make sure model directory exists
        model_dir = os.path.dirname(os.path.abspath(model_path))
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        # Train a pose classifier
        train_pose_classifier(dataset_path, model_path, reference_angles_path=references_path)
    
    elif mode == 'classify' and input_path and model_path:
        # Classify a pose using the trained model
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return
            
        predicted_pose, confidence = classify_pose(input_path, model_path)
        
        # If we have reference angles for the predicted pose, analyze it further
        if predicted_pose in ground_truth_angles:
            print(f"Analyzing pose accuracy against reference for {predicted_pose}...")
            keypoints, pose_angles, comparison_results = analyze_yoga_pose(
                input_path, predicted_pose, angle_tolerance)
            
            if keypoints is not None:
                fig = visualize_pose_analysis(
                    input_path, keypoints, pose_angles, comparison_results, 
                    predicted_pose, angle_tolerance)
                
                # Save the visualization if output path specified
                if output_path:
                    # Create output directory if needed
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    else:
        # parser.print_help()
        print("\nInvalid arguments. Please specify a valid mode and required parameters.")

# Run the main function when the script is executed directly
if __name__ == "__main__":
    import time  # Import here to avoid potential circular imports
    main()