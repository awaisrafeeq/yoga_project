# test_fixed_backend.py
import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

async def test_frame_processing():
    try:
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 100), 20, (255, 255, 255), -1)  # head
        cv2.line(img, (320, 120), (320, 300), (255, 255, 255), 3)  # body
        
        _, buffer = cv2.imencode('.jpg', img)
        test_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        uri = "wss://129442b010ee.ngrok-free.app/ws/pose-analysis"
        
        async with websockets.connect(uri) as websocket:
            print("Connected")
            
            # Wait for connection
            await websocket.recv()
            
            # Start analysis
            await websocket.send(json.dumps({
                "type": "start_analysis",
                "pose_name": "Tree",
                "tolerance": 10.0
            }))
            
            # Get session ID
            response = await websocket.recv()
            data = json.loads(response)
            session_id = data.get('session_id')
            print(f"Session started: {session_id}")
            
            # Send frame
            await websocket.send(json.dumps({
                "type": "pose_frame",
                "data": test_frame,
                "session_id": session_id
            }))
            print("Frame sent, waiting for response...")
            
            # Wait for result
            response = await asyncio.wait_for(websocket.recv(), timeout=15)
            data = json.loads(response)
            print(f"Response type: {data.get('type')}")
            
            if data.get('type') == 'pose_analysis':
                print("SUCCESS! Backend is working")
                print(f"Pose detected: {data.get('pose_detected')}")
                print(f"Message: {data.get('message', 'N/A')}")
            else:
                print(f"Unexpected response: {data}")
                
    except Exception as e:
        print(f"Test failedfffffffffff: {e}")

asyncio.run(test_frame_processing())