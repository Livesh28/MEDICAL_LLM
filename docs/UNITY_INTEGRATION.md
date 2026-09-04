# Unity VR ↔ FastAPI Integration Guide

## Overview
This document specifies how the Unity C# VR client communicates asynchronously with the FastAPI server at `http://127.0.0.1:8000/ask`.

---

## 1. C# Unity Script (`VRVoiceAssistant.cs`)

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

[System.Serializable]
public class VRAskPayload {
    public string question;
    public int current_step;
    public string step_name;
    public string last_mistake;
    public int top_k_chunks = 2;
    public float temperature = 0.3f;
}

[System.Serializable]
public class VRSourceMetadata {
    public string source_id;
    public string title;
    public string section;
    public string page;
    public string url;
}

[System.Serializable]
public class VRAskResponse {
    public string question;
    public string answer;
    public string engine;
    public bool grounded;
    public string confidence;
    public VRSourceMetadata[] sources;
}

public class VRVoiceAssistant : MonoBehaviour {
    [SerializeField] private string serverUrl = "http://127.0.0.1:8000/ask";

    public void QueryAIVoiceAssistant(string traineeSpeechText, int currentStep, string stepName, string lastMistake) {
        StartCoroutine(PostQuestionCoroutine(traineeSpeechText, currentStep, stepName, lastMistake));
    }

    private IEnumerator PostQuestionCoroutine(string questionText, int currentStep, string stepName, string lastMistake) {
        VRAskPayload payload = new VRAskPayload {
            question = questionText,
            current_step = currentStep,
            step_name = stepName,
            last_mistake = lastMistake
        };
        
        string jsonPayload = JsonUtility.ToJson(payload);

        using (UnityWebRequest www = new UnityWebRequest(serverUrl, "POST")) {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success) {
                VRAskResponse res = JsonUtility.FromJson<VRAskResponse>(www.downloadHandler.text);
                Debug.Log($"[VR AI Response] Engine: {res.engine} | Answer: {res.answer}");
                // Send res.answer text to Neural TTS & play via VR Headset AudioSource
            } else {
                Debug.LogError($"[VR AI Network Error]: {www.error}");
            }
        }
    }
}
```

---

## 2. Deterministic VR StepManager Immutability Rule
* The FastAPI response text is played through the VR headset audio player.
* The response payload **NEVER** mutates or overrides C# `StepManager`, `Veni`, `StepList`, or `SnapZone` state.
