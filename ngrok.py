# test_ngrok.py
import asyncio
import websockets
import json

async def test_ngrok_websocket():
    uri = "wss://38a19dec60ef.ngrok-free.app/ws/pose-analysis"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to ngrok WebSocket!")
        
        response = await websocket.recv()
        print(f"Received: {response}")
        
        # Start analysis
        await websocket.send(json.dumps({
            "type": "start_analysis", 
            "pose_name": "Tree",
            "tolerance": 10.0
        }))
        
        response = await websocket.recv()
        print(f"Analysis started: {response}")

asyncio.run(test_ngrok_websocket())