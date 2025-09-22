import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import pickle
import json
from datetime import datetime
import mediapipe as mp
 
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the MediaPipe pose keypoint names and indices
# MediaPipe uses different keypoints than COCO
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", 
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", 
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", 
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

# Create a mapping from keypoint indices to names
KEYPOINT_DICT = {i: name for i, name in enumerate(KEYPOINT_NAMES)}

# Define connections between keypoints for visualization
# These connections represent the skeleton structure for MediaPipe pose
SKELETON = [
    # Face connections
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # Upper body connections
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Torso connections
    (11, 23), (12, 24), (23, 24),
    # Legs connections
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32)
]

# Define angles to calculate (triplets of keypoints forming angles)
# Format: [joint_idx, central_idx, joint_idx, angle_name]
# The central_idx is the keypoint at which we measure the angle
# Updated for MediaPipe keypoint indices
ANGLES_TO_CALCULATE = [
    [13, 11, 23, "Left Shoulder Angle"],    # Left Shoulder Angle (elbow-shoulder-hip)
    [24, 12, 14, "Right Shoulder Angle"],   # Right Shoulder Angle (hip-shoulder-elbow)
    [11, 13, 15, "Left Elbow Angle"],       # Left Elbow Angle (shoulder-elbow-wrist)
    [12, 14, 16, "Right Elbow Angle"],      # Right Elbow Angle (shoulder-elbow-wrist)
    [23, 25, 27, "Left Knee Angle"],        # Left Knee Angle (hip-knee-ankle)
    [24, 26, 28, "Right Knee Angle"],       # Right Knee Angle (hip-knee-ankle)
    [11, 23, 25, "Left Hip Angle"],         # Left Hip Angle (shoulder-hip-knee)
    [12, 24, 26, "Right Hip Angle"],        # Right Hip Angle (shoulder-hip-knee)
]

# Colors for visualization (one distinct color per keypoint)
POSE_PALETTE = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], 
    [230, 230, 0], [255, 153, 255], [153, 204, 255], 
    [255, 102, 255], [255, 51, 255], [102, 178, 255], 
    [51, 153, 255], [255, 153, 153], [255, 102, 102], 
    [255, 51, 51], [153, 255, 153], [102, 255, 102], 
    [51, 255, 51], [0, 255, 0], [0, 255, 128], [0, 255, 255],
    [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 51, 255],
    [0, 0, 255], [51, 0, 255], [102, 0, 255], [153, 0, 255],
    [204, 0, 255], [255, 0, 255], [255, 0, 204], [255, 0, 153],
    [255, 0, 102], [255, 0, 51]
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

def mediapipe_to_keypoints(results):
    """
    Convert MediaPipe pose results to a numpy array of keypoints.
    
    Args:
        results: MediaPipe pose detection results
        
    Returns:
        Numpy array of keypoints with shape [1, 33, 3] where each keypoint is [x, y, visibility]
    """
    if not results.pose_landmarks:
        return np.zeros((1, 33, 3))
    
    # Initialize output array for one person (MediaPipe typically detects one person)
    keypoints = np.zeros((1, 33, 3))
    
    # Extract keypoints from the detection
    pose_landmarks = results.pose_landmarks.landmark
    
    # Convert to numpy array format
    for i, landmark in enumerate(pose_landmarks):
        keypoints[0, i, 0] = landmark.x  # x coordinate (normalized 0-1)
        keypoints[0, i, 1] = landmark.y  # y coordinate (normalized 0-1)
        keypoints[0, i, 2] = landmark.visibility  # visibility (confidence)
    
    return keypoints

def denormalize_keypoints(keypoints, image_width, image_height):
    """
    Convert normalized keypoints (0-1) to pixel coordinates.
    
    Args:
        keypoints: Numpy array of normalized keypoints with shape [n, k, 3]
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Numpy array of denormalized keypoints in pixel coordinates
    """
    denormalized = keypoints.copy()
    
    # Scale x and y coordinates to pixel values
    denormalized[:, :, 0] *= image_width
    denormalized[:, :, 1] *= image_height
    
    return denormalized

def calculate_pose_angles(keypoints, confidence_threshold=0.5):
    """
    Calculate a set of predefined angles for pose analysis.
    
    Args:
        keypoints: Numpy array of keypoints with shape [n, 33, 3] where each keypoint is [x, y, visibility]
        confidence_threshold: Minimum confidence to consider a keypoint valid
        
    Returns:
        Dictionary of angles for each person detected
    """
    results = []
    
    # For each person in the image
    for person_idx in range(keypoints.shape[0]):
        person_angles = {}
        kpts = keypoints[person_idx]
        
        # Calculate each predefined angle
        for a_idx, center_idx, c_idx, angle_name in ANGLES_TO_CALCULATE:
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

def compare_with_ground_truth(calculated_angles, ground_truth, tolerance=15.0):
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
    
    # Initialize the MediaPipe pose detector
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
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
                # Read the image and process with MediaPipe
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  Could not read image: {img_file}")
                    continue
                    
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                
                # Process the image with MediaPipe
                results = pose.process(image_rgb)
                
                if not results.pose_landmarks:
                    print(f"  No pose detected in: {img_file}")
                    continue
                
                # Convert MediaPipe results to keypoints format
                keypoints = mediapipe_to_keypoints(results)
                
                # Denormalize keypoints to pixel coordinates for visualization
                keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
                
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
    
    # Close the MediaPipe pose model
    pose.close()
    
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
        keypoints: Numpy array of keypoints with shape [n, 33, 3] where each keypoint is [x, y, visibility]
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

def analyze_yoga_pose(image_path, pose_name=None, angle_tolerance=15.0):
    """
    Analyze a yoga pose image using MediaPipe and compare with ground truth if available.
    
    Args:
        image_path: Path to the image file
        pose_name: Name of the yoga pose (for ground truth comparison)
        angle_tolerance: Acceptable difference in degrees
        
    Returns:
        Tuple of (keypoints, pose_angles, comparison_results)
    """
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, None, None
            
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        # Process the image with MediaPipe
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print(f"No pose detected in {image_path}")
            return None, None, None
        
        # Convert MediaPipe results to keypoints format
        keypoints = mediapipe_to_keypoints(results)
        
        # Denormalize keypoints to pixel coordinates for visualization
        keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
        
        # Calculate pose angles
        pose_angles = calculate_pose_angles(keypoints_denorm)
        
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
        
        return keypoints_denorm, pose_angles, comparison_results

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
        
        # Initialize MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                return None, 0
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            
            # Process the image with MediaPipe
            results = pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print("No pose detected in the image.")
                return None, 0
            
            # Convert MediaPipe results to keypoints format
            keypoints = mediapipe_to_keypoints(results)
            
            # Denormalize keypoints to pixel coordinates for angle calculation
            keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
            
            # Calculate pose angles
            pose_angles = calculate_pose_angles(keypoints_denorm)
            
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

def process_video(video_path, pose_name=None, angle_tolerance=15.0, output_path=None, model_path=None):
    """
    Process a video file for yoga pose analysis with enhanced visual feedback.
    
    Args:
        video_path: Path to the video file
        pose_name: Name of the yoga pose (for ground truth comparison)
        angle_tolerance: Acceptable difference in degrees
        output_path: Path to save the processed video
        model_path: Path to the classifier model for pose recognition
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
    
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize video writer if output_path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
    
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
            
            # Load reference angles from model file if available
            global ground_truth_angles
            if 'reference_angles' in model_data and model_data['reference_angles']:
                ground_truth_angles = model_data['reference_angles']
                print(f"Loaded reference angles for {len(ground_truth_angles)} poses from model file")
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            classifier = None
    
    # Check if ground truth exists for this pose
    can_compare = pose_name in ground_truth_angles if pose_name else False
    gt_angles = ground_truth_angles.get(pose_name, {}) if pose_name else {}
    
    if pose_name:
        if can_compare:
            print(f"Ground truth angles loaded for pose: {pose_name}")
        else:
            print(f"Warning: No ground truth angles found for pose: {pose_name}")
            print(f"Available poses: {list(ground_truth_angles.keys())}")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Use medium complexity for video
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    detected_pose = None
    pose_confidence = 0
    start_time = time.time()
    processing_times = []
    
    print("Starting video processing...")
    
    while cap.isOpened():
        frame_start_time = time.time()
        success, frame = cap.read()
        if not success:
            break
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape
        
        # Process the frame with MediaPipe
        results = pose.process(image_rgb)
        
        # Make a copy for drawing
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        if results.pose_landmarks:
            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Convert MediaPipe results to keypoints format
            keypoints = mediapipe_to_keypoints(results)
            
            # Denormalize keypoints to pixel coordinates
            keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
            
            # Calculate pose angles
            pose_angles = calculate_pose_angles(keypoints_denorm)
            
            if pose_angles and len(pose_angles) > 0:
                # Add angle annotations for key joints (similar to image processing)
                for angle_def in ANGLES_TO_CALCULATE:
                    p1_idx, center_idx, p3_idx, angle_name = angle_def
                    
                    # Get keypoint positions
                    x1, y1, conf1 = keypoints_denorm[0, p1_idx]
                    x2, y2, conf2 = keypoints_denorm[0, center_idx]
                    x3, y3, conf3 = keypoints_denorm[0, p3_idx]
                    
                    # Check if we have a valid angle
                    if (angle_name in pose_angles[0] and 
                        pose_angles[0][angle_name] is not None and
                        conf1 > 0.5 and conf2 > 0.5 and conf3 > 0.5):
                        
                        angle_value = pose_angles[0][angle_name]
                        
                        # Determine if this angle is within tolerance (if comparison possible)
                        within_tolerance = True
                        if can_compare and angle_name in gt_angles:
                            gt_value = gt_angles[angle_name]
                            if gt_value is not None:
                                within_tolerance = abs(angle_value - gt_value) <= angle_tolerance
                        
                        # Select text color based on whether angle is within tolerance
                        text_color = (0, 255, 0) if within_tolerance else (0, 0, 255)  # Green or Red
                        
                        # Place the angle text near the joint
                        text_x = int(x2 + 10)
                        text_y = int(y2 - 10)
                        
                        # Add background rectangle for better text visibility
                        text_size = cv2.getTextSize(f'{angle_value:.1f}°', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (text_x - 2, text_y - text_size[1] - 2),
                                    (text_x + text_size[0] + 2, text_y + 2),
                                    (255, 255, 255), -1)
                        
                        cv2.putText(annotated_frame, f'{angle_value:.1f}°', 
                                  (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                        
                        # Draw a red circle around incorrect angles
                        if not within_tolerance and can_compare and angle_name in gt_angles:
                            cv2.circle(annotated_frame, (int(x2), int(y2)), 25, (0, 0, 255), 3)
                
                # Classify the pose if we have a classifier and not a fixed pose
                if classifier is not None and not pose_name and frame_count % 15 == 0:  # Classify every 15 frames
                    try:
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
                            # Format for display
                            display_name = ' '.join(word.capitalize() for word in predicted_class.replace('_', ' ').replace('-', ' ').split())
                            detected_pose = display_name
                            pose_confidence = confidence
                            
                            # Try to find matching reference angles
                            matched_name, ref_angles, found_match = find_pose_in_references(
                                predicted_class, ground_truth_angles
                            )
                            
                            if found_match:
                                can_compare = True
                                gt_angles = ref_angles
                                pose_name = matched_name  # Use the detected pose for comparison
                    except Exception as e:
                        print(f"Error during pose classification: {e}")
                
                # Compare with ground truth if available
                comparison_results = None
                if can_compare and gt_angles:
                    comparison_results = compare_with_ground_truth(pose_angles[0], gt_angles, angle_tolerance)
                    
                    # Generate correction feedback
                    feedback = provide_correction_feedback(comparison_results, angle_tolerance)
                    
                    # Display feedback on the image (top 3 corrections)
                    y_pos = 30
                    for i, msg in enumerate(feedback[:3]):
                        # Add background rectangle for better text visibility
                        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (8, y_pos - text_size[1] - 2),
                                    (text_size[0] + 12, y_pos + 2),
                                    (0, 0, 0), -1)  # Black background
                        
                        cv2.putText(annotated_frame, msg, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        y_pos += 35
                    
                    # Calculate accuracy
                    correct_count = sum(1 for res in comparison_results.values() if res["within_tolerance"])
                    total_count = sum(1 for res in comparison_results.values() if res["calculated"] is not None)
                    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
                    
                    # Display accuracy on the image
                    accuracy_text = f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})"
                    accuracy_color = (0, 255, 0) if accuracy >= 75 else (0, 165, 255) if accuracy >= 50 else (0, 0, 255)
                    
                    # Add background for accuracy text
                    text_size = cv2.getTextSize(accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_frame, 
                                (width - text_size[0] - 12, 20),
                                (width - 2, 20 + text_size[1] + 8),
                                (0, 0, 0), -1)  # Black background
                    
                    cv2.putText(annotated_frame, accuracy_text, 
                              (width - text_size[0] - 10, 20 + text_size[1] + 4), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, accuracy_color, 2)
            
            # Display pose name if available
            pose_text = ""
            if pose_name:
                pose_text = f"Pose: {pose_name}"
            elif detected_pose:
                pose_text = f"Detected: {detected_pose} ({pose_confidence*100:.1f}%)"
            
            if pose_text:
                # Add background for pose text
                text_size = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, 
                            (8, height - 25),
                            (text_size[0] + 12, height - 5),
                            (0, 0, 0), -1)  # Black background
                
                cv2.putText(annotated_frame, pose_text, 
                          (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # If no pose detected
            no_pose_text = "No pose detected"
            text_size = cv2.getTextSize(no_pose_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((width - text_size[0]) / 2)
            text_y = int(height / 2)
            
            # Add background
            cv2.rectangle(annotated_frame, 
                        (text_x - 10, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 10, text_y + 5),
                        (0, 0, 0), -1)
            
            cv2.putText(annotated_frame, no_pose_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate and display frame rate
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        processing_times.append(frame_processing_time)
        
        # Calculate current FPS
        current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
        
        # Display FPS on the frame
        fps_text = f"FPS: {current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_frame, 
                    (width - text_size[0] - 12, height - 30),
                    (width - 2, height - 5),
                    (0, 0, 0), -1)  # Black background
        
        cv2.putText(annotated_frame, fps_text, 
                  (width - text_size[0] - 10, height - 12), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow text
        
        # Display frame number
        frame_text = f"Frame: {frame_count + 1}/{total_frames}"
        cv2.putText(annotated_frame, frame_text, 
                  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display progress
        frame_count += 1
        if frame_count % 30 == 0 or frame_count == 1:  # Update every 30 frames
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Processing: {progress:.1f}% complete (Frame {frame_count}/{total_frames}, Avg FPS: {avg_fps:.2f})")
        
        # Write the frame to output video if writer is initialized
        if writer:
            writer.write(annotated_frame)
        
        # Optional: Display the frame in real-time (comment out for faster processing)
        # cv2.imshow('Video Processing', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pose.close()
    
    # Print final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    print(f"\nVideo processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print(f"Average frame processing time: {avg_processing_time:.4f} seconds")
    
    if output_path:
        print(f"Output video saved to: {output_path}")

def normalize_keypoints(keypoints, confidence_threshold=0.5):
    """
    Normalize keypoints to make them invariant to scale and position.
    
    Args:
        keypoints: Numpy array of keypoints with shape [n, 33, 3] where each keypoint is [x, y, visibility]
        confidence_threshold: Minimum confidence to consider a keypoint valid
        
    Returns:
        Normalized keypoints with the same shape
    """
    normalized_keypoints = np.copy(keypoints)
    
    # For each person in the image
    for person_idx in range(keypoints.shape[0]):
        kpts = keypoints[person_idx]
        
        # Find the center of the torso (midpoint between shoulders and hips)
        # Using MediaPipe indices: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        left_shoulder = kpts[11]
        right_shoulder = kpts[12]
        left_hip = kpts[23]
        right_hip = kpts[24]
        
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
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # Get all pose class folders
    pose_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_idx, pose_class in enumerate(pose_classes):
        class_dir = os.path.join(dataset_path, pose_class)
        pose_name = pose_class.replace('_', ' ').title()
        print(f"Processing class {class_idx+1}/{len(pose_classes)}: {pose_name}")
        
        # Get all image files in this class
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Read the image and process with MediaPipe
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not read image: {img_path}")
                    continue
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                
                # Process the image with MediaPipe
                results = pose.process(image_rgb)
                
                if not results.pose_landmarks:
                    print(f"No pose detected in: {img_file}")
                    continue
                
                # Convert MediaPipe results to keypoints format
                keypoints = mediapipe_to_keypoints(results)
                
                # Denormalize keypoints to pixel coordinates for angle calculation
                keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
                
                # Calculate pose angles
                pose_angles = calculate_pose_angles(keypoints_denorm)
                
                if pose_angles and len(pose_angles) > 0:
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
    
    # Close the MediaPipe pose detector
    pose.close()
    
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

def visualize_pose_analysis(image_path, keypoints, pose_angles, comparison_results=None, pose_name=None, angle_tolerance=15.0):
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
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
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

def real_time_pose_analysis(pose_name=None, angle_tolerance=15.0, mirror_mode=True, model_path=None, references_path=None):
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
        cap = cv2.VideoCapture(0)  # Try camera 1 first
        if not cap.isOpened():
            print("Could not open camera 1, trying camera 0")
            cap = cv2.VideoCapture(0)  # Fall back to camera 0
            
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
        
        # Initialize MediaPipe Pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use medium complexity for real-time
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create a window
        cv2.namedWindow('MediaPipe Pose Analysis', cv2.WINDOW_NORMAL)
        
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
            image_height, image_width, _ = display_frame.shape
            
            try:
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(image_rgb)
                
                # Reset person detection status for this frame
                person_detected = False
                
                # Check if any pose was detected
                if results.pose_landmarks:
                    # Person is detected
                    person_detected = True
                    
                    # Draw pose landmarks on the frame
                    mp_drawing.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Convert MediaPipe results to keypoints format
                    keypoints = mediapipe_to_keypoints(results)
                    
                    # Denormalize keypoints to pixel coordinates
                    keypoints_denorm = denormalize_keypoints(keypoints, image_width, image_height)
                    
                    # Calculate pose angles
                    pose_angles = calculate_pose_angles(keypoints_denorm)
                    
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
                        
                        # Add angle annotations for key joints
                        for angle_def in ANGLES_TO_CALCULATE:
                            if len(angle_def) < 4:
                                continue  # Skip invalid angle definitions
                            
                            p1_idx, center_idx, p3_idx, angle_name = angle_def
                            
                            if p1_idx >= keypoints_denorm.shape[1] or center_idx >= keypoints_denorm.shape[1] or p3_idx >= keypoints_denorm.shape[1]:
                                continue  # Skip if indices are out of bounds
                            
                            # Get keypoint positions from the denormalized keypoints
                            x1, y1, conf1 = keypoints_denorm[0, p1_idx]
                            x2, y2, conf2 = keypoints_denorm[0, center_idx]
                            x3, y3, conf3 = keypoints_denorm[0, p3_idx]
                            
                            # Only calculate angle if all keypoints have sufficient confidence
                            if conf1 > 0.5 and conf2 > 0.5 and conf3 > 0.5:
                                # Check if this angle exists in the calculated angles
                                if (angle_name in pose_angles[0] and 
                                    pose_angles[0][angle_name] is not None):
                                    
                                    angle_value = pose_angles[0][angle_name]
                                    
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
            cv2.imshow('MediaPipe Pose Analysis', display_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
    
    except Exception as e:
        print(f"Critical error in real_time_pose_analysis: {e}")
        # Try to clean up resources
        try:
            if 'cap' in locals() and cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            if 'pose' in locals() and pose is not None:
                pose.close()
        except:
            pass
            

# def main():
#     """
#     Main function to demonstrate the yoga pose analysis system.
#     """
#     import os
#     # import argparse
    
#     # # Set up command line argument parsing
#     # parser = argparse.ArgumentParser(description='Yoga Pose Analysis with MediaPipe')
#     # parser.add_argument('--mode', type=str, 
#     #                   choices=['image', 'video', 'webcam', 'train', 'classify', 'generate-references'],
#     #                   default='webcam', help='Operation mode')
#     # parser.add_argument('--input', type=str, help='Input image or video path')
#     # parser.add_argument('--output', type=str, help='Output path for processed video or files')
#     # parser.add_argument('--pose', type=str, help='Yoga pose name for ground truth comparison')
#     # parser.add_argument('--dataset', type=str, help='Dataset path for training')
#     # parser.add_argument('--expert-dataset', type=str, help='Path to expert demonstrations for reference angle generation')
#     # parser.add_argument('--model', type=str, default='model/yoga_pose_model.pkl', help='Model path for saving/loading')
#     # parser.add_argument('--references', type=str, help='Path to reference angles file')
#     # parser.add_argument('--save-keypoints', action='store_true', help='Save detected keypoints for reference poses')
#     # parser.add_argument('--tolerance', type=float, default=15.0, help='Angle tolerance in degrees')
    
#     # args = parser.parse_args()
    
#     # For direct execution without args, can uncomment and modify these lines
#     mode = 'video'
#     input = "/kaggle/input/pose-video/Seated Forward Bend - Trim.mp4"
#     output = 'SFB MP.mp4'
#     pose = None
#     model = "/kaggle/working/pose correction.pkl"
#     references = "/kaggle/working/reference_data/reference_angles_20250529_092217.pkl"
#     save_keypoints = True
#     tolerance = 15.0
    
#     # Load reference angles if specified
#     global ground_truth_angles
#     if references and os.path.exists(references):
#         print(f"Loading reference angles from {references}")
#         ground_truth_angles = load_reference_angles(references)
    
#     if mode == 'generate-references' and expert_dataset:
#         # Generate reference angles from expert demonstrations
#         output_dir = output if output else 'reference_data'
#         os.makedirs(output_dir, exist_ok=True)
#         ground_truth_angles = generate_reference_angles(
#             expert_dataset, 
#             output_directory=output_dir,
#             save_keypoints=save_keypoints
#         )
    
#     elif mode == 'image' and input:
#         # Process a single image
#         keypoints, pose_angles, comparison_results = analyze_yoga_pose(input, pose, tolerance)
#         if keypoints is not None:
#             fig = visualize_pose_analysis(
#                 input, keypoints, pose_angles, comparison_results, 
#                 pose, tolerance
#             )
#             if output:
#                 # Create output directory if needed
#                 output_dir = os.path.dirname(output)
#                 if output_dir and not os.path.exists(output_dir):
#                     os.makedirs(output_dir, exist_ok=True)
#                 fig.savefig(output, dpi=300, bbox_inches='tight')
        
#         # If model_path exists, also classify the pose
#         if model and os.path.exists(model):
#             predicted_pose, confidence = classify_pose(input, model)
            
#             # If we have reference angles for the predicted pose, analyze it further
#             if predicted_pose in ground_truth_angles:
#                 print(f"Analyzing pose accuracy against reference for {predicted_pose}...")
#                 keypoints, pose_angles, comparison_results = analyze_yoga_pose(
#                     input, predicted_pose, tolerance)
                
#                 if keypoints is not None:
#                     fig = visualize_pose_analysis(
#                         input, keypoints, pose_angles, comparison_results, 
#                         predicted_pose, tolerance)
                    
#                     # Save the visualization if output path specified
#                     if output:
#                         # Create output directory if needed
#                         output_dir = os.path.dirname(output)
#                         if output_dir and not os.path.exists(output_dir):
#                             os.makedirs(output_dir, exist_ok=True)
#                         # Save with a modified name to avoid overwriting
#                         base, ext = os.path.splitext(output)
#                         classified_output = f"{base}_classified{ext}"
#                         fig.savefig(classified_output, dpi=300, bbox_inches='tight')
    
#     elif mode == 'video' and input:
#         # Process a video file
#         if output:
#             # Create output directory if needed
#             output_dir = os.path.dirname(output)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir, exist_ok=True)
#         process_video(input, pose, tolerance, output, model)
    
#     elif mode == 'webcam':
#         # Real-time analysis using webcam
#         real_time_pose_analysis(
#             pose_name=pose, 
#             angle_tolerance=tolerance, 
#             mirror_mode=True, 
#             model_path=model, 
#             references_path=references
#         )
    
#     elif mode == 'train' and dataset:
#         # Make sure model directory exists
#         model_dir = os.path.dirname(os.path.abspath(model))
#         if model_dir:
#             os.makedirs(model_dir, exist_ok=True)
            
#         # Train a pose classifier
#         train_pose_classifier(dataset, model, reference_angles_path=references)
    
#     elif mode == 'classify' and input and model:
#         # Classify a pose using the trained model
#         if not os.path.exists(model):
#             print(f"Error: Model file {model} not found")
#             return
            
#         predicted_pose, confidence = classify_pose(input, model)
        
#         # If we have reference angles for the predicted pose, analyze it further
#         if predicted_pose in ground_truth_angles:
#             print(f"Analyzing pose accuracy against reference for {predicted_pose}...")
#             keypoints, pose_angles, comparison_results = analyze_yoga_pose(
#                 input, predicted_pose, tolerance)
            
#             if keypoints is not None:
#                 fig = visualize_pose_analysis(
#                     input, keypoints, pose_angles, comparison_results, 
#                     predicted_pose, tolerance)
                
#                 # Save the visualization if output path specified
#                 if output:
#                     # Create output directory if needed
#                     output_dir = os.path.dirname(output)
#                     if output_dir and not os.path.exists(output_dir):
#                         os.makedirs(output_dir, exist_ok=True)
#                     fig.savefig(output, dpi=300, bbox_inches='tight')
    
#     else:
#         parser.print_help()
#         print("\nInvalid arguments. Please specify a valid mode and required parameters.")

# # Run the main function when the script is executed directly
# if __name__ == "__main__":
#     main()