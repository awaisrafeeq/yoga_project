def main():
    if model == 'yolo':
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
        # global ground_truth_angles
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
                            predicted_pose, angle_tolerance=15.0)
                        
                        # Save the visualization if output path specified
                        if output_path:
                            # Create output directory if needed
                            output_dir = os.path.dirname(output_path)
                            if output_dir and not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        elif mode == 'video' and input_path:
            # Process a video file
            if output_path:
                # Create output directory if needed
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            process_video(input_path, pose_name, angle_tolerance, output_path)
        
        elif mode == 'webcam':
            # Real-time analysis using webcam
            # real_time_pose_analysis(pose_name, angle_tolerance=15.0, model_path=model_path, references_path=references_path)
            real_time_pose_analysis(pose_name=None, angle_tolerance=15.0, mirror_mode=True, model_path=model_path, references_path=references_path)
        
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

    elif model == 'mediapipe':
        # Load reference angles if specified
        # global ground_truth_angles
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
            keypoints, pose_angles, comparison_results = analyze_yoga_pose(input_path, pose_name, angle_tolerance)
            if keypoints is not None:
                fig = visualize_pose_analysis(
                    input_path, keypoints, pose_angles, comparison_results, 
                    pose_name, angle_tolerance
                )
                if output_path:
                    # Create output directory if needed
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # If model_path exists, also classify the pose
            if model and os.path.exists(model):
                predicted_pose, confidence = classify_pose(input_path, model)
                
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
                            # Save with a modified name to avoid overwriting
                            base, ext = os.path.splitext(output_path)
                            classified_output = f"{base}_classified{ext}"
                            fig.savefig(classified_output, dpi=300, bbox_inches='tight')
        
        elif mode == 'video' and input_path:
            # Process a video file
            if output_path:
                # Create output directory if needed
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            process_video(input_path, pose_name, angle_tolerance, output_path, model)
        
        elif mode == 'webcam':
            # Real-time analysis using webcam
            real_time_pose_analysis(
                pose_name=pose_name, 
                angle_tolerance=angle_tolerance, 
                mirror_mode=True, 
                model_path=model, 
                references_path=references_path
            )
        
        elif mode == 'train' and dataset_path:
            # Make sure model directory exists
            model_dir = os.path.dirname(os.path.abspath(model))
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
                
            # Train a pose classifier
            train_pose_classifier(dataset_path, model, reference_angles_path=references_path)
        
        elif mode == 'classify' and input_path and model:
            # Classify a pose using the trained model
            if not os.path.exists(model):
                print(f"Error: Model file {model} not found")
                return
                
            predicted_pose, confidence = classify_pose(input_path, model)
            
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
            print("\nInvalid arguments. Please specify a valid mode and required parameters.")


if __name__ == '__main__':
    model = 'yolo' # or mediapipe
    mode = 'video' # [train, generate-references, classify, image, video, webcam]
    input_path = r"Tree Pose video - Trim.mp4"
    output_path = None
    pose_name = None
    dataset_path = None
    expert_dataset = None
    model_path = r"pose correction.pkl"
    references_path = r"reference_angles_20250520_093142.pkl"
    save_keypoints = True
    angle_tolerance = 15.0

    if model == 'yolo':
        from Yoga_pose_estimation_YOLO import *
    elif model == 'mediapipe':
        from Yoga_pose_estimation_mediapipe import *

    main()