import asyncio
import websockets
import json
import cv2
import base64
import numpy as np
import time

async def test_websocket_debug():
    """Debug WebSocket connection with detailed logging"""
    print("=== Debug WebSocket Test ===")
    
    try:
        # Create test image
        img = cv2.rectangle(
            np.zeros((480, 640, 3), dtype=np.uint8), 
            (100, 100), (500, 400), (255, 255, 255), -1
        )
        _, buffer = cv2.imencode('.jpg', img)
        test_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        uri = "ws://localhost:8000/ws/pose-analysis"
        print(f"Connecting to {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Wait for connection message
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📨 Received: {data}")
            
            # Start analysis
            start_message = {
                "type": "start_analysis",
                "pose_name": "Tree",
                "tolerance": 10.0
            }
            await websocket.send(json.dumps(start_message))
            print("🎯 Sent start analysis")
            
            # Wait for analysis started
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📨 Received: {data}")
            session_id = data.get('session_id')
            
            if session_id:
                # Send frame
                frame_message = {
                    "type": "pose_frame",
                    "data": test_frame,
                    "session_id": session_id
                }
                await websocket.send(json.dumps(frame_message))
                print("📷 Sent test frame")
                
                # Wait for pose analysis with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    data = json.loads(response)
                    print(f"📨 Received: {data}")
                    
                    if data.get('type') == 'pose_analysis':
                        print("✅ Frame processing successful!")
                        print(f"   - Pose detected: {data.get('pose_detected')}")
                        if data.get('pose_detected'):
                            print(f"   - Pose name: {data.get('pose_name')}")
                            print(f"   - Confidence: {data.get('confidence')}")
                        else:
                            print(f"   - Message: {data.get('message')}")
                    elif data.get('type') == 'error':
                        print(f"❌ Error: {data.get('message')}")
                        
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for frame processing result")
                
                # Stop analysis
                stop_message = {
                    "type": "stop_analysis",
                    "session_id": session_id
                }
                await websocket.send(json.dumps(stop_message))
                print("🛑 Sent stop analysis")
            
            else:
                print("❌ No session ID received")
                
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

def test_direct_api():
    """Test the process_frame function directly via HTTP"""
    print("\n=== Testing Direct API Call ===")
    import requests
    
    try:
        # Create test image
        img = cv2.rectangle(
            np.zeros((480, 640, 3), dtype=np.uint8), 
            (100, 100), (500, 400), (255, 255, 255), -1
        )
        _, buffer = cv2.imencode('.jpg', img)
        test_frame = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        # Start session
        response = requests.post("http://localhost:8000/api/session/start", 
                               json={"pose_name": "Tree", "tolerance": 10.0})
        session_data = response.json()
        session_id = session_data['session_id']
        print(f"✅ Session started: {session_id}")
        
        # Test if we can call process_frame directly (we'll need to add this endpoint)
        print("ℹ️  Note: Direct frame processing would require additional HTTP endpoint")
            
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")

if __name__ == "__main__":
    print("🔍 Starting WebSocket Debug Test")
    asyncio.run(test_websocket_debug())
    test_direct_api()