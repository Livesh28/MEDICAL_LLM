using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace MedicalVR.Voice
{
    public enum SimulationMode
    {
        TrainingMode,
        TestMode
    }

    [System.Serializable]
    public class AskPayload
    {
        public string question;
        public int current_step;
        public string step_name;
        public string last_mistake;
        public int top_k_chunks = 2;
        public float temperature = 0.3f;
    }

    [System.Serializable]
    public class SourceMetadata
    {
        public string source_id;
        public string title;
        public string section;
        public string page;
        public string url;
    }

    [System.Serializable]
    public class AskResponse
    {
        public string question;
        public string answer;
        public string engine;
        public bool grounded;
        public string confidence;
        public string intent;
        public SourceMetadata[] sources;
    }

    /// <summary>
    /// Phase 9 & 17 Module: Unity VR Voice Assistant Orchestration Script
    /// Coordinates STT, /ask query execution, TTS playback, and UI updates.
    /// Strictly guarantees StepManager immutability and disables voice during Test Mode.
    /// </summary>
    public class VRVoiceAssistant : MonoBehaviour
    {
        [Header("Backend Endpoints")]
        [SerializeField] private string sttServerUrl = "http://127.0.0.1:8000/stt";
        [SerializeField] private string askServerUrl = "http://127.0.0.1:8000/ask";
        [SerializeField] private float requestTimeoutSec = 10f;

        [Header("VR Simulation Core References")]
        [SerializeField] private SimulationMode currentMode = SimulationMode.TrainingMode;
        [SerializeField] private int currentStep = 0;
        [SerializeField] private string stepName = "Wash Hands";
        [SerializeField] private string lastMistake = "None";

        [Header("Subsystem References")]
        [SerializeField] private VoiceInputManager voiceInputManager;
        [SerializeField] private TTSManager ttsManager;
        [SerializeField] private VRVoiceUIManager uiManager;

        public event Action<string> OnTranscriptReceived;
        public event Action<AskResponse> OnResponseReceived;
        public event Action<string> OnErrorOccurred;

        private void OnEnable()
        {
            if (voiceInputManager != null)
            {
                voiceInputManager.OnRecordingStarted += HandleRecordingStarted;
                voiceInputManager.OnRecordingStopped += HandleRecordingStopped;
                voiceInputManager.OnRecordingError += HandleRecordingError;
            }
        }

        private void OnDisable()
        {
            if (voiceInputManager != null)
            {
                voiceInputManager.OnRecordingStarted -= HandleRecordingStarted;
                voiceInputManager.OnRecordingStopped -= HandleRecordingStopped;
                voiceInputManager.OnRecordingError -= HandleRecordingError;
            }
        }

        public void SetSimulationMode(SimulationMode mode)
        {
            currentMode = mode;
            if (currentMode == SimulationMode.TestMode)
            {
                Debug.Log("[VRVoiceAssistant] Voice Assistant DISABLED in Test Mode.");
                if (uiManager != null) uiManager.SetVoiceDisabledUI("Voice Assistant is inactive during Test Mode.");
            }
        }

        public void UpdateStepContext(int stepIndex, string name, string mistake)
        {
            // Update local snapshot only; StepManager remains authoritative
            currentStep = stepIndex;
            stepName = name;
            lastMistake = mistake;
        }

        private void HandleRecordingStarted()
        {
            if (currentMode == SimulationMode.TestMode) return;
            if (uiManager != null) uiManager.SetListeningState();
        }

        private void HandleRecordingStopped(byte[] wavBytes)
        {
            if (currentMode == SimulationMode.TestMode) return;
            StartCoroutine(ProcessAudioToSTT(wavBytes));
        }

        private void HandleRecordingError(string error)
        {
            if (uiManager != null) uiManager.SetErrorState(error);
            OnErrorOccurred?.Invoke(error);
        }

        private IEnumerator ProcessAudioToSTT(byte[] audioBytes)
        {
            if (uiManager != null) uiManager.SetThinkingState("Transcribing voice...");

            WWWForm form = new WWWForm();
            form.AddBinaryData("file", audioBytes, "trainee_speech.wav", "audio/wav");

            using (UnityWebRequest www = UnityWebRequest.Post(sttServerUrl, form))
            {
                www.timeout = (int)requestTimeoutSec;
                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    string jsonResult = www.downloadHandler.text;
                    string transcript = ParseTranscriptFromJson(jsonResult);

                    if (!string.IsNullOrWhiteSpace(transcript))
                    {
                        Debug.Log($"[VRVoiceAssistant] Recognized Speech: '{transcript}'");
                        if (uiManager != null) uiManager.UpdateTranscriptText(transcript);
                        OnTranscriptReceived?.Invoke(transcript);

                        // Proceed to /ask query processing
                        QueryAIVoiceAssistant(transcript);
                    }
                    else
                    {
                        HandleRecordingError("Could not transcribe speech clearly.");
                    }
                }
                else
                {
                    HandleRecordingError("Voice assistant is temporarily unavailable.");
                }
            }
        }

        public void QueryAIVoiceAssistant(string questionText)
        {
            if (currentMode == SimulationMode.TestMode)
            {
                Debug.LogWarning("[VRVoiceAssistant] Query ignored: Voice assistant is disabled in Test Mode.");
                return;
            }
            StartCoroutine(PostAskRequest(questionText));
        }

        private IEnumerator PostAskRequest(string questionText)
        {
            if (uiManager != null) uiManager.SetThinkingState("Consulting Clinical AI...");

            AskPayload payload = new AskPayload
            {
                question = questionText,
                current_step = currentStep,
                step_name = stepName,
                last_mistake = lastMistake
            };

            string jsonPayload = JsonUtility.ToJson(payload);

            using (UnityWebRequest www = new UnityWebRequest(askServerUrl, "POST"))
            {
                byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
                www.uploadHandler = new UploadHandlerRaw(bodyRaw);
                www.downloadHandler = new DownloadHandlerBuffer();
                www.SetRequestHeader("Content-Type", "application/json");
                www.timeout = (int)requestTimeoutSec;

                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    AskResponse res = JsonUtility.FromJson<AskResponse>(www.downloadHandler.text);
                    if (res != null && !string.IsNullOrWhiteSpace(res.answer))
                    {
                        Debug.Log($"[VR AI Response] Engine: {res.engine} | Intent: {res.intent} | Answer: {res.answer}");
                        if (uiManager != null) uiManager.SetAnswerState(res.answer);
                        OnResponseReceived?.Invoke(res);

                        // Trigger TTS Playback
                        if (ttsManager != null)
                        {
                            ttsManager.SpeakText(res.answer);
                        }
                    }
                    else
                    {
                        HandleRecordingError("Received empty response from assistant.");
                    }
                }
                else
                {
                    Debug.LogError($"[VR AI Network Error]: {www.error}");
                    HandleRecordingError("Voice assistant is temporarily unavailable.");
                }
            }
        }

        private string ParseTranscriptFromJson(string json)
        {
            try
            {
                int index = json.IndexOf("\"transcript\":\"");
                if (index != -1)
                {
                    int start = index + 14;
                    int end = json.IndexOf("\"", start);
                    return json.Substring(start, end - start);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ParseTranscript] Error: {ex.Message}");
            }
            return "";
        }
    }
}
