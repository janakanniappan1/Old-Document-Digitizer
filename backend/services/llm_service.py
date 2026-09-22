import os
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self):
        # Cloud AI Providers (100% Free Tiers for 24/7 Cloud Hosting)
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        # Local Ollama Configuration
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.preferred_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self._client = None
        self.model = self.preferred_model

        if self.groq_api_key:
            self.provider = "Groq"
            self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info("Using Groq Cloud AI (%s) for text correction", self.model)
        elif self.gemini_api_key:
            self.provider = "Gemini"
            self.model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            logger.info("Using Google Gemini Cloud AI (%s) for text correction", self.model)
        else:
            self.provider = "Ollama"
            try:
                import ollama
                self._client = ollama.Client(host=self.ollama_host)
                self.model = self._resolve_ollama_model()
                self._warmup_model()
            except Exception as e:
                logger.warning("Ollama init notice: %s", e)

    def _resolve_ollama_model(self):
        if not self._client:
            return self.preferred_model
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
        if not self._client:
            return
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

        # 1. Cloud Provider: Groq API
        if self.groq_api_key:
            return self._correct_with_groq(text_clean)

        # 2. Cloud Provider: Gemini API
        if self.gemini_api_key:
            return self._correct_with_gemini(text_clean)

        # 3. Local Provider: Ollama
        return self._correct_with_ollama(text_clean)

    def _correct_with_groq(self, text):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OldDocumentDigitizer/1.0"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert OCR correction assistant. Fix OCR errors, broken words, and misspelled characters in historical or degraded documents. Preserve all names, dates, numbers, and layout. Return ONLY the repaired document text with no introduction, markdown wrappers, or explanation."
                    },
                    {
                        "role": "user",
                        "content": self.build_prompt(text)
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["choices"][0]["message"]["content"].strip()
                return self._clean_llm_response(result, text)
        except Exception as e:
            logger.warning("Groq AI correction failed (falling back to raw text): %s", e)
            return text

    def _correct_with_gemini(self, text):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": self.build_prompt(text)}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1500
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return self._clean_llm_response(result, text)
        except Exception as e:
            logger.warning("Gemini AI correction failed: %s", e)
            return text

    def _correct_with_ollama(self, text):
        if not self._client:
            return text

        word_count = len(text.split())
        token_limit = min(1200, max(64, int(word_count * 2.2)))

        try:
            response = self._client.generate(
                model=self.model,
                prompt=self.build_prompt(text),
                stream=False,
                keep_alive="24h",
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_thread": 8,
                    "num_predict": token_limit
                }
            )

            answer = (
                response.get("response", "") if isinstance(response, dict)
                else getattr(response, "response", "")
            ).strip()

            return self._clean_llm_response(answer, text)

        except Exception as e:
            logger.warning("Ollama correction failed: %s", e)
            return text

    def _clean_llm_response(self, answer, original):
        if not answer:
            return original

        # Strip markdown code block wrapper if present
        if answer.startswith("```") and answer.endswith("```"):
            lines = answer.splitlines()
            if len(lines) >= 2:
                answer = "\n".join(lines[1:-1]).strip()

        # Remove conversational prefixes
        for prefix in [
            "Here is the corrected text:",
            "Corrected Output:",
            "Here is the corrected version:",
            "Output:"
        ]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        return answer if answer else original

    # ==========================================
    # Check Connectivity
    # ==========================================
    def check_server(self):
        if self.groq_api_key or self.gemini_api_key:
            return True
        if self._client:
            try:
                self._client.list()
                return True
            except Exception:
                return False
        return False
