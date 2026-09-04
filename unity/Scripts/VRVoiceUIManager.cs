using UnityEngine;
using UnityEngine.UI;

namespace MedicalVR.Voice
{
    /// <summary>
    /// Phase 11 Module: VR Voice Assistant UI Overlay Manager
    /// Displays Push-to-Talk button, Listening status, recognized transcript,
    /// Thinking animation indicator, generated answer text, and error states in VR.
    /// </summary>
    public class VRVoiceUIManager : MonoBehaviour
    {
        [Header("UI Canvas Components")]
        [SerializeField] private Text statusText;
        [SerializeField] private Text transcriptText;
        [SerializeField] private Text answerText;
        [SerializeField] private GameObject listeningIndicator;
        [SerializeField] private GameObject thinkingIndicator;
        [SerializeField] private Button pushToTalkButton;

        public void SetListeningState()
        {
            if (statusText) statusText.text = "Listening...";
            if (listeningIndicator) listeningIndicator.SetActive(true);
            if (thinkingIndicator) thinkingIndicator.SetActive(false);
            if (transcriptText) transcriptText.text = "";
            if (answerText) answerText.text = "";
        }

        public void UpdateTranscriptText(string transcript)
        {
            if (transcriptText) transcriptText.text = $"\"{transcript}\"";
        }

        public void SetThinkingState(string message = "Thinking...")
        {
            if (statusText) statusText.text = message;
            if (listeningIndicator) listeningIndicator.SetActive(false);
            if (thinkingIndicator) thinkingIndicator.SetActive(true);
        }

        public void SetAnswerState(string answer)
        {
            if (statusText) statusText.text = "Answer:";
            if (listeningIndicator) listeningIndicator.SetActive(false);
            if (thinkingIndicator) thinkingIndicator.SetActive(false);
            if (answerText) answerText.text = answer;
        }

        public void SetErrorState(string errorMessage)
        {
            if (statusText) statusText.text = "Status: Error";
            if (listeningIndicator) listeningIndicator.SetActive(false);
            if (thinkingIndicator) thinkingIndicator.SetActive(false);
            if (answerText) answerText.text = errorMessage;
        }

        public void SetVoiceDisabledUI(string message)
        {
            if (statusText) statusText.text = message;
            if (listeningIndicator) listeningIndicator.SetActive(false);
            if (thinkingIndicator) thinkingIndicator.SetActive(false);
            if (answerText) answerText.text = "";
            if (pushToTalkButton) pushToTalkButton.interactable = false;
        }
    }
}
