#!/usr/bin/env python3
"""
Phase 3 Module: STT Benchmark Script
Evaluates Whisper STT accuracy, latency, and failure rate across 20 audio test queries.
Generates outputs/stt_test_report.json.
"""

import os
import sys
import json
import time
import numpy as np
import wave
import tempfile
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.stt_service import WhisperSTTService

TEST_SPOKEN_QUERIES = [
    "What should I do next?",
    "Why do we clean the site with alcohol?",
    "Why was that wrong?",
    "What is the maximum time a tourniquet can remain on?",
    "What angle should the needle be inserted?",
    "Repeat the instruction",
    "Where is the annotator guidance?",
    "What object do I grab?",
    "What is the patient's blood pressure?",
    "What medication does the patient take?",
    "Tell me about venipuncture",
    "Why must alcohol dry for 30 seconds?",
    "What tube is drawn first for blood culture?",
    "What causes hemolysis during blood draw?",
    "Why did I get an error?",
    "Help me I am stuck",
    "Where do I place the tourniquet?",
    "Explain phlebotomy safety",
    "What is the function of sodium citrate in blue tubes?",
    "Why must tubes be inverted gently?"
]

def generate_synthetic_speech_wav(text: str) -> bytes:
    """
    Generates synthetic speech audio WAV bytes using macOS 'say' command or synthetic tone fallback.
    """
    tmp_aiff = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_aiff.close()
    tmp_wav.close()

    try:
        import subprocess
        subprocess.run(["say", "-v", "Samantha", "-r", "160", text, "-o", tmp_aiff.name], check=True)
        # Convert AIFF to WAV using afconvert or ffmpeg
        subprocess.run(["afconvert", "-f", "WAVE", "-c", "1", "-d", "LEI16@16000", tmp_aiff.name, tmp_wav.name], check=True)
        
        with open(tmp_wav.name, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    except Exception:
        # Fallback synthetic PCM WAV generator
        sample_rate = 16000
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.5
        audio = (tone * 32767).astype(np.int16)

        byte_io = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(byte_io.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        with open(byte_io.name, "rb") as f:
            wav_bytes = f.read()
        os.remove(byte_io.name)
        return wav_bytes
    finally:
        for p in [tmp_aiff.name, tmp_wav.name]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

def calculate_word_accuracy(reference: str, hypothesis: str) -> float:
    ref_words = [w.strip(".,?!;").lower() for w in reference.split() if w.strip(".,?!;")]
    hyp_words = [w.strip(".,?!;").lower() for w in hypothesis.split() if w.strip(".,?!;")]
    
    if not ref_words:
        return 1.0 if not hyp_words else 0.0
    
    matches = sum(1 for w in ref_words if w in hyp_words)
    return matches / len(ref_words)

def run_stt_benchmark():
    print(f"[+] Starting Whisper STT Benchmark across {len(TEST_SPOKEN_QUERIES)} Spoken Queries...")
    stt = WhisperSTTService(model_name="tiny")
    
    results = []
    total_latency_ms = 0.0
    accuracies = []
    failed_count = 0

    for idx, prompt_text in enumerate(TEST_SPOKEN_QUERIES):
        audio_bytes = generate_synthetic_speech_wav(prompt_text)
        res = stt.transcribe_audio_bytes(audio_bytes)
        
        transcript = res.get("transcript", "")
        stt_ms = res.get("stt_ms", 0.0)
        status = res.get("status", "error")
        
        acc = calculate_word_accuracy(prompt_text, transcript) if transcript else 0.0
        accuracies.append(acc)
        total_latency_ms += stt_ms
        
        if status != "success" or not transcript:
            failed_count += 1

        results.append({
            "query_id": idx + 1,
            "spoken_prompt": prompt_text,
            "recognized_transcript": transcript,
            "accuracy": round(acc, 4),
            "stt_latency_ms": stt_ms,
            "status": status
        })
        print(f"  [{idx+1}/{len(TEST_SPOKEN_QUERIES)}] Speech: '{prompt_text}' -> STT: '{transcript}' ({stt_ms} ms)")

    avg_latency = total_latency_ms / len(TEST_SPOKEN_QUERIES)
    mean_accuracy = float(np.mean(accuracies))

    report = {
        "total_spoken_queries": len(TEST_SPOKEN_QUERIES),
        "failed_transcriptions": failed_count,
        "mean_word_accuracy": round(mean_accuracy, 4),
        "avg_stt_latency_ms": round(avg_latency, 2),
        "test_results": results
    }

    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/stt_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Whisper STT Benchmark Complete.")
    print(f"    - Mean Word Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"    - Failed Transcriptions: {failed_count}")
    print(f"    - Avg Latency: {avg_latency:.2f} ms")
    print(f"    - Report Saved To: {report_path}")
    return report

if __name__ == "__main__":
    run_stt_benchmark()
