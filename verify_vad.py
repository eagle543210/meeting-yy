
import asyncio
import numpy as np
import time
from services.voiceprint_service import VoiceprintService
from services.milvus_service import MilvusManager
from services.mongodb_manager import MongoDBManager
from core.speech_to_text.stt_processor import SpeechToTextProcessor
from config.settings import settings

async def verify_realtime_vad():
    # Mocking managers
    class MockManager:
        def __getattr__(self, name): return lambda *args, **kwargs: None
        @property
        def is_connected(self): return True
        @property
        def db(self): return {}

    mock_milvus = MockManager()
    mock_mongo = MockManager()
    
    vp_service = VoiceprintService(settings, mock_milvus, mock_mongo)
    await vp_service.load_model()
    
    stt_processor = SpeechToTextProcessor(settings)
    # No need to load STT model for logic test, mock process_utterance if needed
    
    # 1. Simulate speech chunks (0.5s each)
    sample_rate = 16000
    chunk_size = int(0.5 * sample_rate)
    speech_chunk = np.random.uniform(-0.5, 0.5, chunk_size).astype(np.float32)
    silence_chunk = np.zeros(chunk_size, dtype=np.float32)
    
    meeting_id = "test_meeting"
    
    print("--- Testing Silence Trigger ---")
    # Send 3 speech chunks (1.5s)
    for i in range(3):
        res = await vp_service.process_realtime_audio(speech_chunk, sample_rate, meeting_id, stt_processor)
        print(f"Chunk {i+1} (Speech): results={len(res)}")
        time.sleep(0.1)
    
    # Send silence chunks (total > 1.0s to trigger)
    for i in range(3):
        res = await vp_service.process_realtime_audio(silence_chunk, sample_rate, meeting_id, stt_processor)
        print(f"Chunk {i+1} (Silence): results={len(res)}")
        if res: break
        time.sleep(0.5)

    print("\n--- Testing Timeout Trigger (15s) ---")
    # Send 32 speech chunks (~16s)
    results_count = 0
    for i in range(32):
        res = await vp_service.process_realtime_audio(speech_chunk, sample_rate, meeting_id, stt_processor)
        if res:
            print(f"Chunk {i+1} (Speech): TRIGGERED! results={len(res)}")
            results_count += 1
        else:
            if i % 5 == 0: print(f"Chunk {i+1} (Speech): buffer size={len(vp_service.speech_buffer[meeting_id])}")
    
    if results_count > 0:
        print("✅ Timeout trigger works!")
    else:
        print("❌ Timeout trigger failed.")

if __name__ == "__main__":
    asyncio.run(verify_realtime_vad())
