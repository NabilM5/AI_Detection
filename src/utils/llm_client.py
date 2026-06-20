import json
import os
import urllib.error
import urllib.request

from openai import OpenAI


class LLMAPIError(RuntimeError):
    def __init__(self, provider, status_code, message, retry_after=None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after


PLACEHOLDER_KEYS = {
    "",
    "your_key_here",
    "your-real-api-key-here",
    "sk-or-v1-your-openrouter-key-here",
    "sk-your-openai-key-here",
    "sk-ant-your-anthropic-key-here",
    "your-google-ai-studio-key-here",
}


def _get_required_env(name):
    value = os.getenv(name, "").strip()

    if value in PLACEHOLDER_KEYS:
        raise RuntimeError(
            f"{name} is not set to a real key. Edit .env and set {name}=your_real_key."
        )

    return value


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
        self.api_key = _get_required_env("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

        if not self.model:
            raise RuntimeError("LLM_MODEL is not set. Edit .env and choose a model.")

        if self.provider == "openai_compatible":
            self._validate_openai_compatible_config()
            self.client = self._create_openai_compatible_client()
        elif self.provider == "anthropic":
            self.client = None
        elif self.provider == "google":
            self.client = None
        else:
            raise RuntimeError(
                "Unsupported LLM_PROVIDER. Use openai_compatible, anthropic, or google."
            )

    def _validate_openai_compatible_config(self):
        if "openrouter.ai" in self.base_url and not self.api_key.startswith("sk-or-"):
            raise RuntimeError(
                "Your .env is configured for OpenRouter, but LLM_API_KEY does not look "
                "like an OpenRouter key.\n\n"
                "Use one of these valid configurations:\n\n"
                "OpenRouter:\n"
                "LLM_PROVIDER=openai_compatible\n"
                "LLM_API_KEY=sk-or-v1-...\n"
                "LLM_MODEL=openrouter/free\n"
                "LLM_BASE_URL=https://openrouter.ai/api/v1\n\n"
                "OpenAI:\n"
                "LLM_PROVIDER=openai_compatible\n"
                "LLM_API_KEY=sk-proj-...\n"
                "LLM_MODEL=gpt-4o-mini\n"
                "LLM_BASE_URL=https://api.openai.com/v1"
            )

        if "api.openai.com" in self.base_url and self.model.startswith("openrouter/"):
            raise RuntimeError(
                "Your .env is configured for OpenAI, but LLM_MODEL is an OpenRouter model.\n"
                "For OpenAI use a model like gpt-4o-mini, not openrouter/free."
            )

    def _create_openai_compatible_client(self):
        base_url = self.base_url or None
        site_url = os.getenv("LLM_SITE_URL", "").strip()
        app_name = os.getenv("LLM_APP_NAME", "").strip()

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-OpenRouter-Title"] = app_name

        return OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers=headers or None,
            timeout=self.timeout,
        )

    def generate_text(self, prompt, temperature=0.7):
        if self.provider == "openai_compatible":
            return self._generate_openai_compatible(prompt, temperature)

        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, temperature)

        if self.provider == "google":
            return self._generate_google(prompt, temperature)

        raise RuntimeError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _generate_openai_compatible(self, prompt, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, prompt, temperature):
        payload = {
            "model": self.model,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "700")),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise LLMAPIError("anthropic", e.code, error_body) from e

        text_parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()

    def _generate_google(self, prompt, temperature):
        base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url.rstrip('/')}/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": int(os.getenv("LLM_MAX_TOKENS", "700")),
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise LLMAPIError(
                "google",
                e.code,
                error_body,
                retry_after=self._extract_google_retry_after(error_body),
            ) from e

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Google Gemini API returned no candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if "text" in part]
        return "\n".join(text_parts).strip()

    def _extract_google_retry_after(self, error_body):
        try:
            data = json.loads(error_body)
        except json.JSONDecodeError:
            return None

        details = data.get("error", {}).get("details", [])
        for detail in details:
            retry_delay = detail.get("retryDelay")
            if retry_delay and retry_delay.endswith("s"):
                try:
                    return float(retry_delay[:-1])
                except ValueError:
                    return None

        return None
