from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import base64
import pickle
import os
import uuid
from datetime import datetime
import asyncio
import json
import logging
from io import BytesIO
from PIL import Image
import uvicorn

# Import your existing yoga pose analysis functions
from Yoga_pose_estimation_YOLO import (
    calculate_pose_angles,
    compare_with_ground_truth,
    provide_correction_feedback,
    find_pose_in_references,
    load_reference_angles,
    YOLO,
    ANGLES_TO_CALCULATE,
    SKELETON,
    POSE_PALETTE
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Yoga Pose Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
yolo_model = None
classifier_model = None
pose_classes = []
angle_features = []
reference_angles = {}
active_sessions = {}
active_connections: Dict[str, WebSocket] = {}

# Configuration
MODEL_PATH = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\pose correction.pkl"
REFERENCES_PATH = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\angles_final.pkl"
YOLO_MODEL_PATH = r"E:\Ai Data House Intern\nalla-maneendra-ai-full-stack-developer\newmilestonw\yoga\yolo11x-pose.pt"

# Pydantic models
class SessionStartRequest(BaseModel):
    pose_name: Optional[str] = None
    tolerance: float = 10.0

class SessionResponse(BaseModel):
    session_id: str
    settings: Dict[str, Any]

class PoseInfo(BaseModel):
    name: str
    display_name: str
    description: str

class ModelStatus(BaseModel):
    yolo_ready: bool
    classifier_ready: bool
    reference_angles_loaded: bool
    available_poses: int
    reference_poses: int
    timestamp: str

class SettingsUpdate(BaseModel):
    angle_tolerance: Optional[float] = 10.0
    confidence_threshold: Optional[float] = 0.5
    mirror_mode: Optional[bool] = True

class PoseFrameData(BaseModel):
    data: str
    session_id: str
    timestamp: Optional[str] = None

class PoseAnalysisSession:
    def __init__(self, session_id: str, pose_name: Optional[str] = None, tolerance: float = 10.0):
        self.session_id = session_id
        self.pose_name = pose_name
        self.tolerance = tolerance
        self.start_time = datetime.now()
        self.frame_count = 0
        self.total_accuracy = 0
        self.accuracy_count = 0
        self.corrections_given = 0
        self.is_active = True
        
    def add_accuracy_measurement(self, accuracy: float):
        if accuracy is not None:
            self.total_accuracy += accuracy
            self.accuracy_count += 1
    
    def get_average_accuracy(self) -> float:
        if self.accuracy_count > 0:
            return self.total_accuracy / self.accuracy_count
        return 0
    
    def get_summary(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'frames_processed': self.frame_count,
            'average_accuracy': self.get_average_accuracy(),
            'corrections_given': self.corrections_given,
            'pose_name': self.pose_name
        }

async def load_models():
    """Load YOLO model, classifier, and reference angles"""
    global yolo_model, classifier_model, pose_classes, angle_features, reference_angles
    
    try:
        # Load YOLO model
        if os.path.exists(YOLO_MODEL_PATH):
            yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
        else:
            logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
            return False
        
        # Load classifier model
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            classifier_model = model_data.get('model')
            pose_classes = model_data.get('classes', [])
            angle_features = model_data.get('features', [])
            logger.info(f"Classifier model loaded with {len(pose_classes)} poses")
        else:
            logger.warning(f"Classifier model not found at {MODEL_PATH}")
        
        # Load reference angles
        if os.path.exists(REFERENCES_PATH):
            reference_angles = load_reference_angles(REFERENCES_PATH)
            logger.info(f"Reference angles loaded for {len(reference_angles)} poses")
        else:
            logger.warning(f"Reference angles not found at {REFERENCES_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

async def process_frame(frame_data: str, session_id: str) -> Dict[str, Any]:
    """Process a single frame for pose analysis"""
    global yolo_model, classifier_model, reference_angles
    
    logger.info(f"Starting frame processing for session {session_id}")  # Add this line

    
    try:
        # Decode base64 image
        logger.info("Decoding base64 image")  # Add this line

        image_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'error': 'Failed to decode image'}
        
        # Get session info
        session = active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        session.frame_count += 1
        
        # Process with YOLO
        results = yolo_model(frame, verbose=False)
        
        # Check if pose detected
        if len(results) == 0 or not hasattr(results[0], 'keypoints') or len(results[0].keypoints) == 0:
            return {
                'pose_detected': False,
                'message': 'No person detected'
            }
        
        # Extract keypoints
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        if keypoints.shape[0] == 0 or keypoints.shape[1] != 17 or keypoints.shape[2] != 3:
            return {
                'pose_detected': False,
                'message': 'Invalid pose keypoints'
            }
        
        # Calculate pose angles
        pose_angles = calculate_pose_angles(keypoints)
        
        if not pose_angles or not pose_angles[0]:
            return {
                'pose_detected': True,
                'message': 'Pose detected but angles not calculable',
                'keypoints': keypoints[0].tolist()
            }
        
        # Classify pose if classifier available and not in fixed mode
        detected_pose = session.pose_name
        pose_confidence = 0
        current_pose_name = session.pose_name
        can_compare = False
        gt_angles = {}
        
        if not session.pose_name and classifier_model:
            # Auto-classify pose
            valid_angles = sum(1 for angle_name in angle_features
                             if angle_name in pose_angles[0] and pose_angles[0][angle_name] is not None)
            
            if valid_angles >= 4:
                feature_vector = []
                for angle_name in angle_features:
                    angle_value = pose_angles[0].get(angle_name)
                    feature_vector.append(angle_value if angle_value is not None else 0)
                
                feature_vector = np.array(feature_vector).reshape(1, -1)
                
                prediction = classifier_model.predict(feature_vector)[0]
                probabilities = classifier_model.predict_proba(feature_vector)[0]
                confidence = probabilities[prediction]
                
                if confidence >= 0.4:
                    predicted_class = pose_classes[prediction]
                    detected_pose = ' '.join(word.capitalize() for word in predicted_class.replace('_', ' ').split())
                    pose_confidence = confidence
                    
                    # Find matching reference angles
                    matched_name, ref_angles, found_match = find_pose_in_references(
                        predicted_class, reference_angles
                    )
                    
                    if found_match:
                        current_pose_name = matched_name
                        can_compare = True
                        gt_angles = ref_angles
        else:
            # Fixed pose mode
            if session.pose_name:
                matched_name, ref_angles, found_match = find_pose_in_references(
                    session.pose_name, reference_angles
                )
                
                if found_match:
                    current_pose_name = matched_name
                    can_compare = True
                    gt_angles = ref_angles
        
        # Compare with reference if available
        comparison_results = None
        accuracy = None
        corrections = []
        angle_status = {}
        
        if can_compare and gt_angles:
            comparison_results = compare_with_ground_truth(
                pose_angles[0], gt_angles, session.tolerance
            )
            
            corrections = provide_correction_feedback(comparison_results, session.tolerance)
            session.corrections_given += len(corrections)
            
            # Calculate accuracy
            correct_count = sum(1 for res in comparison_results.values()
                              if res["within_tolerance"] and res["ground_truth"] is not None)
            total_count = sum(1 for res in comparison_results.values()
                            if res["calculated"] is not None and res["ground_truth"] is not None)
            
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                session.add_accuracy_measurement(accuracy)
            
            # Create angle status
            for angle_name, result in comparison_results.items():
                angle_status[angle_name] = {
                    'within_tolerance': result['within_tolerance'],
                    'difference': result['difference']
                }
        
        # return {
        #     'pose_detected': True,
        #     'pose_name': detected_pose,
        #     'confidence': float(pose_confidence) if pose_confidence else 0,
        #     'accuracy': accuracy,
        #     'correct_angles': sum(1 for res in comparison_results.values() 
        #                         if res["within_tolerance"]) if comparison_results else 0,
        #     'total_angles': len([res for res in comparison_results.values() 
        #                        if res["calculated"] is not None]) if comparison_results else 0,
        #     'keypoints': keypoints[0].tolist(),
        #     'angles': pose_angles[0],
        #     'corrections': corrections[:3],  # Limit to top 3 corrections
        #     'angle_status': angle_status,
        #     'session_stats': {
        #         'frames_processed': session.frame_count,
        #         'average_accuracy': session.get_average_accuracy()
        #     }
        # }

     
        return {
            'pose_detected': True,
            'pose_name': detected_pose,
            'confidence': float(pose_confidence) if pose_confidence else 0.0,
            'accuracy': float(accuracy) if accuracy else None,
            'correct_angles': int(sum(1 for res in comparison_results.values() 
                                    if res["within_tolerance"]) if comparison_results else 0),
            'total_angles': int(len([res for res in comparison_results.values() 
                                if res["calculated"] is not None]) if comparison_results else 0),
            'keypoints': [[float(x), float(y), float(conf)] for x, y, conf in keypoints[0].tolist()],
            'angles': {k: float(v) if v is not None else None for k, v in pose_angles[0].items()},
            'corrections': corrections[:3],
            'angle_status': {k: {
                'within_tolerance': bool(v['within_tolerance']),
                'difference': float(v['difference']) if v['difference'] is not None else None
            } for k, v in angle_status.items()},
            'session_stats': {
                'frames_processed': int(session.frame_count),
                'average_accuracy': float(session.get_average_accuracy())
            }
        }        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {'error': f'Processing failed: {str(e)}'}

# API Endpoints

@app.post("/api/session/start", response_model=SessionResponse)
async def start_session(request: SessionStartRequest):
    """Initialize pose analysis session"""
    try:
        session_id = str(uuid.uuid4())
        session = PoseAnalysisSession(session_id, request.pose_name, request.tolerance)
        active_sessions[session_id] = session
        
        logger.info(f"Started session {session_id} with pose: {request.pose_name}")
        
        return SessionResponse(
            session_id=session_id,
            settings={
                'pose_name': request.pose_name,
                'tolerance': request.tolerance,
                'timestamp': datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/stop/{session_id}")
async def stop_session(session_id: str):
    """End session and get summary"""
    try:
        session = active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.is_active = False
        summary = session.get_summary()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Stopped session {session_id}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/poses/available", response_model=List[PoseInfo])
async def get_available_poses():
    """Get list of supported poses"""
    try:
        poses = []
        
        # Add poses from classifier
        for pose_class in pose_classes:
            display_name = ' '.join(word.capitalize() for word in pose_class.replace('_', ' ').split())
            poses.append(PoseInfo(
                name=pose_class,
                display_name=display_name,
                description=f'{display_name} yoga pose'
            ))
        
        # Add poses from reference angles that might not be in classifier
        for ref_pose in reference_angles.keys():
            if ref_pose not in [p.display_name for p in poses]:
                poses.append(PoseInfo(
                    name=ref_pose.lower().replace(' ', '_'),
                    display_name=ref_pose,
                    description=f'{ref_pose} yoga pose'
                ))
        
        return poses
        
    except Exception as e:
        logger.error(f"Error getting available poses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/status", response_model=ModelStatus)
async def get_model_status():
    """Check if models are loaded and ready"""
    try:
        return ModelStatus(
            yolo_ready=yolo_model is not None,
            classifier_ready=classifier_model is not None,
            reference_angles_loaded=len(reference_angles) > 0,
            available_poses=len(pose_classes),
            reference_poses=len(reference_angles),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update analysis settings"""
    try:
        updated_settings = {
            'angle_tolerance': settings.angle_tolerance,
            'confidence_threshold': settings.confidence_threshold,
            'mirror_mode': settings.mirror_mode,
            'updated_at': datetime.now().isoformat()
        }
        
        return {'updated_settings': updated_settings}
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/pose-analysis")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pose analysis"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    active_connections[client_id] = websocket
    
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            'type': 'connected',
            'client_id': client_id,
            'message': 'Connected to pose analysis server'
        })
        
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            message_type = data.get('type', 'unknown')
            
            if message_type == 'pose_frame':
                # Process pose frame
                frame_data = data.get('data')
                session_id = data.get('session_id')
                
                if not frame_data or not session_id:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Missing frame data or session_id'
                    })
                    continue
                
                # Check if session exists and is active
                session = active_sessions.get(session_id)
                if not session or not session.is_active:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Invalid or inactive session'
                    })
                    continue
                
                # Process the frame with timeout and error handling
                try:
                    result = await asyncio.wait_for(
                        process_frame(frame_data, session_id), 
                        timeout=10.0  # 10 second timeout
                    )
                    result['type'] = 'pose_analysis'
                    result['timestamp'] = datetime.now().isoformat()
                    
                    # Send result back to client
                    await websocket.send_json(result)
                    
                except asyncio.TimeoutError:
                    logger.error(f"Frame processing timeout for session {session_id}")
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Frame processing timeout - try reducing image quality'
                    })
                except Exception as e:
                    logger.error(f"Frame processing error for session {session_id}: {e}")
                    await websocket.send_json({
                        'type': 'error', 
                        'message': f'Processing failed: {str(e)}'
                    })
                
            elif message_type == 'start_analysis':
                # Start analysis session
                pose_name = data.get('pose_name')
                tolerance = data.get('tolerance', 10.0)
                
                session_id = str(uuid.uuid4())
                session = PoseAnalysisSession(session_id, pose_name, tolerance)
                active_sessions[session_id] = session
                
                await websocket.send_json({
                    'type': 'analysis_started',
                    'session_id': session_id,
                    'pose_name': pose_name,
                    'tolerance': tolerance
                })
                
            elif message_type == 'stop_analysis':
                # Stop analysis session
                session_id = data.get('session_id')
                
                if session_id in active_sessions:
                    session = active_sessions[session_id]
                    session.is_active = False
                    summary = session.get_summary()
                    del active_sessions[session_id]
                    
                    await websocket.send_json({
                        'type': 'analysis_stopped',
                        **summary
                    })
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'Session not found'
                    })
            
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'message': f'Server error: {str(e)}'
        })
    finally:
        # Clean up
        if client_id in active_connections:
            del active_connections[client_id]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(active_sessions),
        'active_connections': len(active_connections)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models when the server starts"""
    logger.info("Initializing yoga pose analysis server...")
    success = await load_models()
    if success:
        logger.info("Server initialization complete")
    else:
        logger.error("Server initialization failed - some features may not work")

if __name__ == '__main__':
    # Run the server
    print("Starting yoga pose analysis server with FastAPI...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )