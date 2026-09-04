using System;
using System.IO;
using UnityEngine;

namespace MedicalVR.Voice
{
    /// <summary>
    /// Phase 2 Module: Unity Microphone Capture Manager
    /// Handles VR microphone recording, push-to-talk triggers, permission checks,
    /// empty-input detection, and WAV PCM byte conversion.
    /// </summary>
    public class VoiceInputManager : MonoBehaviour
    {
        [Header("Microphone Settings")]
        [SerializeField] private string selectedMicrophoneDevice = null;
        [SerializeField] private int sampleRate = 16000;
        [SerializeField] private int maxRecordingDurationSec = 10;
        [SerializeField] private float minVolumeThreshold = 0.01f;

        [Header("Recording State")]
        public bool IsRecording { get; private set; } = false;
        private AudioClip recordedClip = null;

        public event Action OnRecordingStarted;
        public event Action<byte[]> OnRecordingStopped;
        public event Action<string> OnRecordingError;

        private void Start()
        {
            RequestMicrophonePermissions();
            if (Microphone.devices.Length > 0)
            {
                selectedMicrophoneDevice = Microphone.devices[0];
                Debug.Log($"[VoiceInputManager] Using microphone device: {selectedMicrophoneDevice}");
            }
            else
            {
                Debug.LogWarning("[VoiceInputManager] No active microphone devices detected.");
            }
        }

        public void RequestMicrophonePermissions()
        {
#if UNITY_ANDROID
            if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
            {
                UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Microphone);
            }
#endif
        }

        public void StartRecording()
        {
            if (IsRecording) return;

            if (Microphone.devices.Length == 0)
            {
                OnRecordingError?.Invoke("No microphone device connected.");
                return;
            }

            string micName = string.IsNullOrEmpty(selectedMicrophoneDevice) ? Microphone.devices[0] : selectedMicrophoneDevice;
            recordedClip = Microphone.Start(micName, false, maxRecordingDurationSec, sampleRate);
            IsRecording = true;
            OnRecordingStarted?.Invoke();
            Debug.Log($"[VoiceInputManager] Recording started on '{micName}' for max {maxRecordingDurationSec}s.");
        }

        public void StopRecording()
        {
            if (!IsRecording) return;

            string micName = string.IsNullOrEmpty(selectedMicrophoneDevice) ? Microphone.devices[0] : selectedMicrophoneDevice;
            int position = Microphone.GetPosition(micName);
            Microphone.End(micName);
            IsRecording = false;

            if (recordedClip == null || position <= 0)
            {
                OnRecordingError?.Invoke("Recorded audio is empty.");
                return;
            }

            // Extract valid audio samples
            float[] samples = new float[position * recordedClip.channels];
            recordedClip.GetData(samples, 0);

            // Check if audio has sufficient amplitude
            if (CalculateRMSVolume(samples) < minVolumeThreshold)
            {
                OnRecordingError?.Invoke("Speech not detected. Please speak louder.");
                return;
            }

            byte[] wavBytes = EncodeToWAV(samples, recordedClip.channels, sampleRate);
            OnRecordingStopped?.Invoke(wavBytes);
            Debug.Log($"[VoiceInputManager] Recording stopped. Transmitting {wavBytes.Length} bytes.");
        }

        private float CalculateRMSVolume(float[] samples)
        {
            float sum = 0f;
            for (int i = 0; i < samples.Length; i++)
            {
                sum += samples[i] * samples[i];
            }
            return Mathf.Sqrt(sum / samples.Length);
        }

        public static byte[] EncodeToWAV(float[] samples, int channels, int sampleRate)
        {
            using (MemoryStream stream = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                int sampleCount = samples.Length;
                int byteRate = sampleRate * channels * 2;

                // RIFF Header
                writer.Write(new char[4] { 'R', 'I', 'F', 'F' });
                writer.Write(36 + sampleCount * 2);
                writer.Write(new char[4] { 'W', 'A', 'V', 'E' });

                // Subchunk 1 (fmt)
                writer.Write(new char[4] { 'f', 'm', 't', ' ' });
                writer.Write(16); // Subchunk1Size (16 for PCM)
                writer.Write((short)1); // AudioFormat (1 for PCM)
                writer.Write((short)channels);
                writer.Write(sampleRate);
                writer.Write(byteRate);
                writer.Write((short)(channels * 2)); // BlockAlign
                writer.Write((short)16); // BitsPerSample

                // Subchunk 2 (data)
                writer.Write(new char[4] { 'd', 'a', 't', 'a' });
                writer.Write(sampleCount * 2);

                for (int i = 0; i < sampleCount; i++)
                {
                    short sampleInt = (short)Mathf.Clamp(samples[i] * 32767f, -32768f, 32767f);
                    writer.Write(sampleInt);
                }

                writer.Flush();
                return stream.ToArray();
            }
        }
    }
}
