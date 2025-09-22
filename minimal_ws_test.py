# minimal_ws_test_fixed.py
import asyncio
import websockets
import json

async def test_simple_connection():
    try:
        uri = "ws://localhost:8000/ws/pose-analysis"
        print(f"Connecting to {uri}...")
        
        # Remove timeout from connect, add it to recv instead
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Wait for connection message with timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_connection())