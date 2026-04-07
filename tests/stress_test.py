import asyncio
import httpx
import os
import time
import numpy as np
import soundfile as sf
import io

# --- CONFIGURATION ---
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "dev-key-12345")
CONCURRENT_USERS = 3
AUDIO_DURATION_SEC = 10
SAMPLE_RATE = 16000

async def send_audio_request(client, user_id, audio_bytes):
    """Simulates a single user sending an audio file for processing."""
    print(f"🚀 User {user_id}: Sending {AUDIO_DURATION_SEC}s audio to {BASE_URL}/process-audio...")
    
    files = {"file": ("stress_test.wav", audio_bytes, "audio/wav")}
    data = {"difficulty": "Standard Interview", "tier": "Pro (High Spec)"}
    headers = {"X-Internal-Key": INTERNAL_KEY}
    
    start_time = time.time()
    try:
        # Long timeout because Whisper processing (especially Pro tier) can take time under load
        response = await client.post(
            f"{BASE_URL}/process-audio", 
            files=files, 
            data=data, 
            headers=headers,
            timeout=120.0 
        )
        
        duration = time.time() - start_time
        if response.status_code == 200:
            print(f"✅ User {user_id}: Success! (Processed in {duration:.2f}s)")
        else:
            print(f"❌ User {user_id}: Failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"💥 User {user_id}: Request failed: {e}")

async def main():
    print("--- 🎙️ AI Coach System Stress Test ---")
    print(f"Simulating {CONCURRENT_USERS} users simultaneously...")
    
    # 1. Generate a dummy 10-second audio file in memory
    print(f"Creating {AUDIO_DURATION_SEC}s dummy audio buffer...")
    t = np.linspace(0, AUDIO_DURATION_SEC, int(SAMPLE_RATE * AUDIO_DURATION_SEC))
    # Simple sine wave to ensure it's not "silent" (though the app handles silence, we want processing load)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t) 
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV')
    audio_bytes = buffer.getvalue()
    
    # 2. Run concurrent requests
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(CONCURRENT_USERS):
            tasks.append(send_audio_request(client, i+1, audio_bytes))
        
        print("Waiting for all users to start...")
        start_all = time.time()
        await asyncio.gather(*tasks)
        
    print(f"\n--- Stress Test Complete ---")
    print(f"Total Wall Time: {time.time() - start_all:.2f}s")
    print("Check your Streamlit sidebar now for VRAM spikes!")

if __name__ == "__main__":
    asyncio.run(main())
