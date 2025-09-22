import requests
import json
import base64
import cv2
import websocket
import threading
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_status():
    """Test model status endpoint"""
    print("=== Testing Model Status ===")
    response = requests.get(f"{BASE_URL}/api/model/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_available_poses():
    """Test available poses endpoint"""
    print("=== Testing Available Poses ===")
    response = requests.get(f"{BASE_URL}/api/poses/available")
    print(f"Status: {response.status_code}")
    poses = response.json()
    print(f"Found {len(poses)} poses:")
    for pose in poses:
        print(f"  - {pose['display_name']} ({pose['name']})")
    print()
    return poses

def test_session_management():
    """Test session start and stop"""
    print("=== Testing Session Management ===")
    
    # Start session
    start_data = {
        "pose_name": "Tree",
        "tolerance": 10.0
    }
    response = requests.post(f"{BASE_URL}/api/session/start", json=start_data)
    print(f"Start Session Status: {response.status_code}")
    session_data = response.json()
    print(f"Session ID: {session_data['session_id']}")
    
    session_id = session_data['session_id']
    
    # Wait a bit
    time.sleep(2)
    
    # Stop session
    response = requests.post(f"{BASE_URL}/api/session/stop/{session_id}")
    print(f"Stop Session Status: {response.status_code}")
    print(f"Session Summary: {json.dumps(response.json(), indent=2)}")
    print()
    
    return session_id

def test_settings():
    """Test settings update"""
    print("=== Testing Settings Update ===")
    settings_data = {
        "angle_tolerance": 15.0,
        "confidence_threshold": 0.6,
        "mirror_mode": True
    }
    response = requests.put(f"{BASE_URL}/api/settings", json=settings_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def create_test_image():
    """Create a simple test image (black rectangle) encoded as base64"""
    # Create a simple black image
    img = cv2.rectangle(
        np.zeros((480, 640, 3), dtype=np.uint8), 
        (100, 100), (500, 400), (128, 128, 128), -1
    )
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def test_websocket():
    """Test WebSocket connection"""
    print("=== Testing WebSocket Connection ===")
    
    messages_received = []
    session_id = None
    
    def on_message(ws, message):
        data = json.loads(message)
        messages_received.append(data)
        print(f"Received: {data.get('type', 'unknown')}")
        if data.get('type') == 'analysis_started':
            nonlocal session_id
            session_id = data.get('session_id')
            print(f"Analysis started with session ID: {session_id}")
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed")
    
    def on_open(ws):
        print("WebSocket connection opened")
        
        # Start analysis
        start_message = {
            "type": "start_analysis",
            "pose_name": "Tree",
            "tolerance": 10.0
        }
        ws.send(json.dumps(start_message))
        
        # Wait a bit then send a test frame
        time.sleep(1)
        
        # Send test frame
        test_image = create_test_image()
        frame_message = {
            "type": "pose_frame",
            "data": test_image,
            "session_id": session_id
        }
        if session_id:
            ws.send(json.dumps(frame_message))
        
        # Wait then stop analysis
        time.sleep(2)
        if session_id:
            stop_message = {
                "type": "stop_analysis",
                "session_id": session_id
            }
            ws.send(json.dumps(stop_message))
        
        # Close after testing
        time.sleep(1)
        ws.close()
    
    # Connect to WebSocket
    ws_url = "ws://localhost:8000/ws/pose-analysis"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket in a thread with timeout
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Wait for completion
    ws_thread.join(timeout=10)
    
    print(f"WebSocket test completed. Received {len(messages_received)} messages")
    for i, msg in enumerate(messages_received):
        print(f"  {i+1}. {msg.get('type', 'unknown')}: {msg.get('message', '')}")
    print()

def run_all_tests():
    """Run all API tests"""
    print("🧘 Starting Yoga Pose API Tests 🧘\n")
    
    try:
        # Test basic endpoints
        test_health()
        test_model_status()
        test_available_poses()
        test_session_management()
        test_settings()
        
        # Test WebSocket (requires websocket-client library)
        try:
            import websocket
            test_websocket()
        except ImportError:
            print("⚠️  Skipping WebSocket test (install websocket-client for WebSocket testing)")
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure your FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    # Import numpy for test image creation
    try:
        import numpy as np
        run_all_tests()
    except ImportError:
        print("Please install required packages:")
        print("pip install requests opencv-python numpy websocket-client")