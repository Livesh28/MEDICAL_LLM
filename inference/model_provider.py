#!/usr/bin/env python3
"""
Model Provider & Routing Abstraction Layer
Provides clean unified interfaces for:
  1. OllamaModelProvider (OpenBioLLM-8B, Llama 3.2 3B)
  2. MedicalTransformerProvider (110M PyTorch MedicalTransformerLM)
  3. ModelRouter (Configurable selection & AUTO fallback chain)
"""

import os
import sys
import json
import requests
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer_lm import MedicalTransformerLM
from tokenizer.tokenizer import MedicalTokenizer
from configs.model_config import ModelConfig
from training.checkpoint import load_checkpoint
from inference.generate import MedicalGenerator


class BaseModelProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.3, **kwargs) -> str:
        pass

    def generate_stream(self, prompt: str, max_tokens: int = 120, temperature: float = 0.3, **kwargs):
        """Default stream implementation yields full generated string in one chunk if provider doesn't support streaming."""
        yield self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass


class OllamaModelProvider(BaseModelProvider):
    """
    Ollama LLM Provider for local GGUF/Ollama models (OpenBioLLM-8B, Llama 3.2 3B).
    """

    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                # Match full name or base tag
                return any(self.model_name in m or m in self.model_name for m in models)
        except Exception:
            pass
        return False

    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.3, **kwargs) -> str:
        try:
            messages = []
            sys_msg = kwargs.get("system_prompt")
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            res = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            if res.status_code == 200:
                return res.json().get("message", {}).get("content", "").strip()
            else:
                raise RuntimeError(f"Ollama API returned HTTP status {res.status_code}: {res.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed for model '{self.model_name}': {e}")

    def generate_stream(self, prompt: str, max_tokens: int = 120, temperature: float = 0.3, **kwargs):
        """Streams tokens progressively from Ollama /api/chat."""
        try:
            messages = []
            sys_msg = kwargs.get("system_prompt")
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "keep_alive": -1,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            res = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=120)
            if res.status_code == 200:
                for line in res.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
            else:
                yield f" [Error: Ollama HTTP {res.status_code}]"
        except Exception as e:
            yield f" [Error: {e}]"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model_name,
            "base_url": self.base_url
        }


class MedicalTransformerProvider(BaseModelProvider):
    """
    PyTorch Provider for the preserved 110M MedicalTransformerLM.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_v3.pt",
        tokenizer_path: str = "tokenizer/artifacts/tokenizer.json",
        device_name: str = "mps"
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() and device_name == "mps" else "cpu")
        self.generator: Optional[MedicalGenerator] = None
        self._initialize()

    def _initialize(self):
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = MedicalTokenizer(self.tokenizer_path)
            cfg = ModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=768,
                num_layers=12,
                num_heads=12,
                context_length=512
            )
            self.model = MedicalTransformerLM(cfg)
            if os.path.exists(self.checkpoint_path):
                try:
                    load_checkpoint(self.checkpoint_path, self.model, device=self.device)
                except Exception as e:
                    print(f"[!] PyTorch Checkpoint load warning: {e}")
            self.generator = MedicalGenerator(self.model, self.tokenizer, self.device)

    def is_available(self) -> bool:
        return self.generator is not None and os.path.exists(self.checkpoint_path)

    def generate(self, prompt: str, max_tokens: int = 120, temperature: float = 0.3, **kwargs) -> str:
        if not self.generator:
            raise RuntimeError("MedicalTransformerLM generator is not initialized.")
        top_k = kwargs.get("top_k", 40)
        top_p = kwargs.get("top_p", 0.9)
        return self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "pytorch",
            "model": "MedicalTransformerLM",
            "checkpoint": self.checkpoint_path,
            "parameters": 110041216
        }


class ModelRouter:
    """
    Centralized Model Router supporting model selection and AUTO fallback.
    """

    def __init__(self, registry_path: str = "config/models.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.providers: Dict[str, BaseModelProvider] = {}
        self._init_providers()

    def _load_registry(self) -> Dict[str, Any]:
        path = self.registry_path
        if not os.path.exists(path):
            path = "configs/models.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # Default fallback registry
        return {
            "openbiollm": {"provider": "ollama", "model": "richardyoung/openbiollm:latest", "role": "production_candidate"},
            "llama32": {"provider": "ollama", "model": "llama3.2:3b", "role": "benchmark_fallback"},
            "medical_transformer_110m": {"provider": "pytorch", "checkpoint": "checkpoints/best_v3.pt", "role": "research_offline", "parameters": 110040000}
        }

    def _init_providers(self):
        # 1. OpenBioLLM Provider
        ob_cfg = self.registry.get("openbiollm", {})
        self.providers["openbiollm"] = OllamaModelProvider(model_name=ob_cfg.get("model", "richardyoung/openbiollm:latest"))

        # 2. Llama 3.2 Provider
        l3_cfg = self.registry.get("llama32", {})
        self.providers["llama32"] = OllamaModelProvider(model_name=l3_cfg.get("model", "llama3.2:3b"))

        # 3. MedicalTransformer 110M PyTorch Provider
        mt_cfg = self.registry.get("medical_transformer_110m", {})
        self.providers["medical_transformer_110m"] = MedicalTransformerProvider(
            checkpoint_path=mt_cfg.get("checkpoint", "checkpoints/best_v3.pt")
        )
        # Alias for backward compatibility
        self.providers["medical_transformer"] = self.providers["medical_transformer_110m"]

    def select_provider(self, key: str = "auto") -> Tuple[BaseModelProvider, str, Dict[str, Any]]:
        """
        Selects provider based on requested key or AUTO fallback logic.
        
        Returns:
            Tuple of (provider_instance, resolved_model_key, metadata)
        """
        norm_key = key.lower().strip()

        if norm_key == "auto":
            # AUTO Strategy: OpenBioLLM -> Llama 3.2 -> MedicalTransformer 110M
            if self.providers["openbiollm"].is_available():
                return self.providers["openbiollm"], "openbiollm", self.providers["openbiollm"].get_metadata()
            elif self.providers["llama32"].is_available():
                return self.providers["llama32"], "llama32", self.providers["llama32"].get_metadata()
            elif self.providers["medical_transformer_110m"].is_available():
                return self.providers["medical_transformer_110m"], "medical_transformer_110m", self.providers["medical_transformer_110m"].get_metadata()
            else:
                # Default to openbiollm if check fails
                return self.providers["openbiollm"], "openbiollm", self.providers["openbiollm"].get_metadata()

        if norm_key in ("openbiollm", "openbiollm-8b"):
            return self.providers["openbiollm"], "openbiollm", self.providers["openbiollm"].get_metadata()

        if norm_key in ("llama32", "llama3.2", "llama3.2:3b"):
            return self.providers["llama32"], "llama32", self.providers["llama32"].get_metadata()

        if norm_key in ("medical_transformer", "medical_transformer_110m", "pytorch", "110m"):
            return self.providers["medical_transformer_110m"], "medical_transformer_110m", self.providers["medical_transformer_110m"].get_metadata()

        # Fallback if unknown key
        return self.providers["openbiollm"], "openbiollm", self.providers["openbiollm"].get_metadata()

    def generate(self, prompt: str, model_key: str = "auto", max_tokens: int = 120, temperature: float = 0.3, **kwargs) -> Tuple[str, Dict[str, Any]]:
        provider, resolved_key, metadata = self.select_provider(model_key)
        text = provider.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return text, metadata
