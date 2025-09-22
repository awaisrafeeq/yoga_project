import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

async def test_complete_flow():
    """Test complete pose analysis flow with WebSocket"""
    print("=== Complete WebSocket Pose Analysis Test ===")
    
    try:
        # Create test image with a person-like shape
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure
        cv2.circle(img, (320, 100), 20, (255, 255, 255), -1)  # head
        cv2.line(img, (320, 120), (320, 300), (255, 255, 255), 3)  # body
        cv2.line(img, (320, 150), (270, 200), (255, 255, 255), 3)  # left arm
        cv2.line(img, (320, 150), (370, 200), (255, 255, 255), 3)  # right arm
        cv2.line(img, (320, 300), (270, 400), (255, 255, 255), 3)  # left leg
        cv2.line(img, (320, 300), (370, 400), (255, 255, 255), 3)  # right leg
        
        # Encode as base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        test_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        print(f"Created test image: {len(test_frame)} characters")
        
        uri = "ws://localhost:8000/ws/pose-analysis"
        
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Step 1: Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Step 1 - Connection: {data['type']}")
            
            # Step 2: Start analysis
            start_message = {
                "type": "start_analysis",
                "pose_name": "Tree",
                "tolerance": 10.0
            }
            await websocket.send(json.dumps(start_message))
            print("Step 2 - Sent start analysis request")
            
            # Step 3: Wait for analysis started
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Step 3 - Analysis started: {data['type']}")
            session_id = data.get('session_id')
            
            if not session_id:
                print("ERROR: No session ID received")
                return
                
            print(f"Session ID: {session_id}")
            
            # Step 4: Send frame for analysis
            frame_message = {
                "type": "pose_frame",
                "data": test_frame,
                "session_id": session_id
            }
            await websocket.send(json.dumps(frame_message))
            print("Step 4 - Sent frame for analysis")
            
            # Step 5: Wait for pose analysis result (with longer timeout)
            try:
                print("Step 5 - Waiting for pose analysis result...")
                response = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                data = json.loads(response)
                print(f"Step 5 - Received: {data['type']}")
                
                if data.get('type') == 'pose_analysis':
                    print("SUCCESS: Frame processing completed!")
                    print(f"  - Pose detected: {data.get('pose_detected')}")
                    print(f"  - Message: {data.get('message', 'N/A')}")
                    if data.get('pose_detected'):
                        print(f"  - Pose name: {data.get('pose_name')}")
                        print(f"  - Confidence: {data.get('confidence', 0):.2f}")
                        print(f"  - Accuracy: {data.get('accuracy')}")
                        corrections = data.get('corrections', [])
                        print(f"  - Corrections: {len(corrections)}")
                        for i, correction in enumerate(corrections[:2]):
                            print(f"    {i+1}. {correction}")
                elif data.get('type') == 'error':
                    print(f"ERROR: {data.get('message')}")
                else:
                    print(f"Unexpected response: {data}")
                    
            except asyncio.TimeoutError:
                print("TIMEOUT: No response received within 20 seconds")
                print("This suggests the frame processing is taking too long or failing silently")
            
            # Step 6: Stop analysis
            stop_message = {
                "type": "stop_analysis",
                "session_id": session_id
            }
            await websocket.send(json.dumps(stop_message))
            print("Step 6 - Sent stop analysis request")
            
            # Optional: Wait for stop confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"Step 7 - Stop confirmed: {data.get('type')}")
            except asyncio.TimeoutError:
                print("Step 7 - Stop confirmation timeout (normal)")
                
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_with_webcam():
    """Test with actual webcam if available"""
    print("\n=== Testing with Real Webcam ===")
    
    try:
        # Try to capture from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not available")
            return
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture webcam frame")
            return
            
        print(f"Captured webcam frame: {frame.shape}")
        
        # Resize frame to reduce processing time
        frame = cv2.resize(frame, (320, 240))
        
        # Encode as base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        webcam_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        uri = "ws://localhost:8000/ws/pose-analysis"
        
        async with websockets.connect(uri) as websocket:
            # Quick flow for webcam test
            await websocket.recv()  # connection
            
            await websocket.send(json.dumps({
                "type": "start_analysis",
                "pose_name": None,  # Auto-detect
                "tolerance": 15.0
            }))
            
            response = await websocket.recv()  # analysis started
            data = json.loads(response)
            session_id = data.get('session_id')
            
            await websocket.send(json.dumps({
                "type": "pose_frame",
                "data": webcam_frame,
                "session_id": session_id
            }))
            
            print("Sent webcam frame, waiting for analysis...")
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=25.0)
                data = json.loads(response)
                
                if data.get('type') == 'pose_analysis':
                    print("WEBCAM SUCCESS!")
                    print(f"  - Pose detected: {data.get('pose_detected')}")
                    if data.get('pose_detected'):
                        print(f"  - Detected pose: {data.get('pose_name')}")
                        print(f"  - Confidence: {data.get('confidence', 0):.2f}")
                else:
                    print(f"Webcam result: {data.get('type')} - {data.get('message')}")
                    
            except asyncio.TimeoutError:
                print("Webcam test timeout")
                
    except Exception as e:
        print(f"Webcam test failed: {e}")

if __name__ == "__main__":
    print("Starting comprehensive WebSocket tests...")
    asyncio.run(test_complete_flow())
    asyncio.run(test_with_webcam())