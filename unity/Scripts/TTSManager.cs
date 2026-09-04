using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace MedicalVR.Voice
{
    /// <summary>
    /// Phase 10 Module: Unity Text-to-Speech (TTS) & Audio Playback Manager
    /// Downloads AIFF/WAV audio bytes from FastAPI /tts endpoint, converts them into AudioClips,
    /// and plays them through the VR Headset AudioSource.
    /// </summary>
    public class TTSManager : MonoBehaviour
    {
        [Header("Audio Output Components")]
        [SerializeField] private AudioSource vrHeadsetAudioSource;
        [SerializeField] private string ttsApiUrl = "http://127.0.0.1:8000/tts";
        [SerializeField] private string voiceName = "Samantha";
        [SerializeField] private int speechRateWPM = 165;

        public bool IsSpeaking => vrHeadsetAudioSource != null && vrHeadsetAudioSource.isPlaying;

        public event Action OnTTSPlaybackStarted;
        public event Action OnTTSPlaybackFinished;
        public event Action<string> OnTTSError;

        private void Awake()
        {
            if (vrHeadsetAudioSource == null)
            {
                vrHeadsetAudioSource = GetComponent<AudioSource>();
                if (vrHeadsetAudioSource == null)
                {
                    vrHeadsetAudioSource = gameObject.AddComponent<AudioSource>();
                }
            }
            vrHeadsetAudioSource.playOnAwake = false;
            vrHeadsetAudioSource.spatialBlend = 0.0f; // 2D Headset audio output
        }

        public void SpeakText(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            StopSpeaking();
            StartCoroutine(FetchAndPlayTTS(text));
        }

        public void StopSpeaking()
        {
            if (vrHeadsetAudioSource != null && vrHeadsetAudioSource.isPlaying)
            {
                vrHeadsetAudioSource.Stop();
                OnTTSPlaybackFinished?.Invoke();
            }
        }

        private IEnumerator FetchAndPlayTTS(string textToSpeak)
        {
            string encodedText = UnityWebRequest.EscapeURL(textToSpeak);
            string requestUrl = $"{ttsApiUrl}?text={encodedText}&voice={voiceName}&rate={speechRateWPM}";

            using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(requestUrl, AudioType.AIFF))
            {
                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                    if (clip != null && vrHeadsetAudioSource != null)
                    {
                        vrHeadsetAudioSource.clip = clip;
                        vrHeadsetAudioSource.Play();
                        OnTTSPlaybackStarted?.Invoke();
                        Debug.Log($"[TTSManager] Playing TTS audio clip for text: '{textToSpeak.Substring(0, Mathf.Min(30, textToSpeak.Length))}...'");

                        yield return new WaitWhile(() => vrHeadsetAudioSource.isPlaying);
                        OnTTSPlaybackFinished?.Invoke();
                    }
                    else
                    {
                        OnTTSError?.Invoke("Failed to instantiate AudioClip from response.");
                    }
                }
                else
                {
                    string errorMsg = $"TTS Request Error: {www.error}";
                    Debug.LogError($"[TTSManager] {errorMsg}");
                    OnTTSError?.Invoke(errorMsg);
                }
            }
        }
    }
}
