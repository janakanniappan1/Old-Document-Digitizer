import os
import logging

import ollama

logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self):
        # Ollama host is configurable via environment variable.
        # Default: http://127.0.0.1:11434 (localhost — do NOT expose externally)
        # Production: set OLLAMA_HOST=http://127.0.0.1:11434 in your environment
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self._client = ollama.Client(host=self.ollama_host)

        # Preferred model — override with OLLAMA_MODEL env var if needed
        self.preferred_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.model = self._resolve_model()
        self._warmup_model()

    def _resolve_model(self):
        try:
            res = self._client.list()
            models_list = (
                getattr(res, "models", [])
                if hasattr(res, "models")
                else (res.get("models", []) if isinstance(res, dict) else [])
            )
            available = []
            for m in models_list:
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if not name and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    available.append(str(name))

            for m in available:
                if m == self.preferred_model or m.startswith(self.preferred_model.split(":")[0]):
                    return m
            if available:
                return available[0]
        except Exception as e:
            logger.warning("Could not query Ollama models: %s", e)
        return self.preferred_model

    def _warmup_model(self):
        try:
            logger.info("Pre-warming Ollama model (%s) in memory...", self.model)
            self._client.generate(
                model=self.model,
                prompt="OK",
                keep_alive="24h",
                options={"num_predict": 1, "num_thread": 8}
            )
            logger.info("Ollama model warmed up and resident in RAM.")
        except Exception as e:
            logger.warning("Ollama warmup notice (non-fatal): %s", e)

    # ==========================================
    # Build Prompt
    # ==========================================
    def build_prompt(self, text):
        prompt = f"""Fix OCR errors, broken words, and misspelled characters in the document text below.
Guidelines:
1. Preserve all proper nouns, names, titles, dates, numbers, and layout.
2. Fix broken words, missing spaces, and clear OCR character substitutions.
3. Return ONLY the repaired document text with no introduction, explanation, or code blocks.

Input:
Nome Jana Kanniappan, Department AIML, Bannari Amman Institue of Techology
Output:
Name: Jana Kanniappan, Department: AIML, Bannari Amman Institute of Technology

Document:
{text}

Output:"""
        return prompt

    # ==========================================
    # AI Correction
    # ==========================================
    def correct_text(self, text):
        if not text or not text.strip():
            return text or ""

        text_clean = text.strip()

        # Limit token generation budget to avoid runaway CPU loops
        word_count = len(text_clean.split())
        token_limit = min(1200, max(64, int(word_count * 2.2)))

        try:
            response = self._client.generate(
                model=self.model,
                prompt=self.build_prompt(text_clean),
                stream=False,
                keep_alive="24h",
                options={
                    "temperature": 0.1,      # Low temperature for deterministic, factual correction
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_thread": 8,         # Maximize CPU core performance
                    "num_predict": token_limit
                }
            )

            answer = (response.get("response", "") if isinstance(response, dict)
                      else getattr(response, "response", "")).strip()

            if not answer:
                return text_clean

            # Strip markdown code block wrapper if LLM returned one
            if answer.startswith("```") and answer.endswith("```"):
                lines = answer.splitlines()
                if len(lines) >= 2:
                    answer = "\n".join(lines[1:-1]).strip()

            # Remove conversational prefix if any slipped through
            for prefix in [
                "Here is the corrected text:",
                "Corrected Output:",
                "Here is the corrected version:",
                "Output:"
            ]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()

            return answer if answer else text_clean

        except Exception as e:
            logger.warning("LLM correction failed (returning raw text): %s", e)
            return text_clean

    # ==========================================
    # Check Ollama connectivity
    # ==========================================
    def check_server(self):
        try:
            self._client.list()
            return True
        except Exception:
            return False
