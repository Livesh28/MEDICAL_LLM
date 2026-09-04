#!/usr/bin/env python3
"""
Phase 3 Module: Whisper STT Service for VR Voice Assistant
Transcribes raw WAV/PCM audio buffers or audio files into transcript text using local Whisper STT.
Tracks transcription latency, duration, and error status.
"""

import os
import sys
import time
import tempfile
from typing import Dict, Any, Tuple, Optional

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

class WhisperSTTService:
    """
    Local Whisper Speech-to-Text transcriber with model caching and fallback support.
    """
    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        if HAS_WHISPER:
            try:
                print(f"[+] Loading Whisper STT Model ({self.model_name})...")
                self.model = whisper.load_model(self.model_name)
                print("[+] Whisper STT Model Loaded Successfully.")
            except Exception as e:
                print(f"[!] Error loading Whisper model: {e}")
                self.model = None

    def transcribe_audio_bytes(self, audio_bytes: bytes, filename_suffix: str = ".wav") -> Dict[str, Any]:
        """
        Transcribes raw audio byte stream and returns transcript, latency_ms, and status.
        """
        start_time = time.time()
        
        if not audio_bytes or len(audio_bytes) < 100:
            return {
                "transcript": "",
                "duration_sec": 0.0,
                "stt_ms": round((time.time() - start_time) * 1000, 2),
                "status": "empty_input",
                "error": "Audio payload is empty or too short."
            }

        tmp_path = None
        try:
            # 1. Parse WAV bytes to float32 numpy array directly
            audio_array = None
            try:
                import io, wave, numpy as np
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    n_channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_data = wf.readframes(n_frames)
                    audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1:
                        audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
                    audio_array = audio_data
            except Exception as parse_err:
                print(f"[!] WAV buffer direct parse warning: {parse_err}")

            if HAS_WHISPER and self.model is not None:
                if audio_array is not None:
                    result = self.model.transcribe(audio_array, fp16=False)
                else:
                    with tempfile.NamedTemporaryFile(suffix=filename_suffix, delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    result = self.model.transcribe(tmp_path, fp16=False)

                transcript = result.get("text", "").strip()
                stt_latency = (time.time() - start_time) * 1000
                
                return {
                    "transcript": transcript,
                    "duration_sec": round(len(audio_bytes) / 32000.0, 2),
                    "stt_ms": round(stt_latency, 2),
                    "status": "success",
                    "error": None
                }
            else:
                stt_latency = (time.time() - start_time) * 1000
                return {
                    "transcript": "",
                    "duration_sec": 0.0,
                    "stt_ms": round(stt_latency, 2),
                    "status": "whisper_unavailable",
                    "error": "Whisper STT model not loaded on server."
                }
        except Exception as e:
            stt_latency = (time.time() - start_time) * 1000
            return {
                "transcript": "",
                "duration_sec": 0.0,
                "stt_ms": round(stt_latency, 2),
                "status": "error",
                "error": str(e)
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def transcribe_text_mock(self, text: str) -> Dict[str, Any]:
        """
        Pass-through utility for testing text-based spoken inputs directly.
        """
        return {
            "transcript": text.strip(),
            "duration_sec": 1.0,
            "stt_ms": 5.0,
            "status": "success",
            "error": None
        }

if __name__ == "__main__":
    stt = WhisperSTTService(model_name="tiny")
    res = stt.transcribe_text_mock("What should I do next?")
    print(f"Mock STT output: {res}")
