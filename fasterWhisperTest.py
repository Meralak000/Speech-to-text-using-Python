from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue, threading

# Audio settings
samplerate = 16000
block_duration = 0.5   # record in 0.5s blocks
chunk_duration = 2.0   # transcribe every 2s
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Load model (use CUDA for RTX 2050)
model = WhisperModel("small", device="cuda", compute_type="float16")

def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    # Copy block to queue
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype="float32",
                        callback=audio_callback,
                        blocksize=frames_per_block):
        print("🎤 Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            # Take 2s worth of audio
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []  # reset

            # Convert to float32 normalized
            audio_data = audio_data.flatten().astype(np.float32)

            # Run Whisper
            segments, _ = model.transcribe(audio_data, language="en", beam_size=1)

            for segment in segments:
                print("> ", segment.text)

# Start threads
threading.Thread(target=recorder, daemon=True).start()
transcriber()
