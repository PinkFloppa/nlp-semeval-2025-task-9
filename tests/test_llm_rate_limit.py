import io
import json
import urllib.error
from pathlib import Path

import pytest

from baseline import llm
from baseline.io import config_at


def test_rate_limit_includes_retries_and_honors_retry_after(monkeypatch):
    settings = config_at(Path(__file__).resolve().parents[1] / "configs/baseline.json")["llm"]
    clock = [0.0]
    starts = []
    monkeypatch.setattr(llm, "_last_api_request", None)
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "8")
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def fake_urlopen(request, timeout):
        starts.append(clock[0])
        if len(starts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too many requests", {"Retry-After": "70"}, None)
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": "milk"}]}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    assert llm.generate("prompt", settings, "test-key")[0] == "milk"
    assert llm.generate("next prompt", settings, "test-key")[0] == "milk"
    assert starts == pytest.approx([0, 70, 77.5])
    assert settings["requests_per_minute"] == 10  # cache identity stays unchanged


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_rate_override(monkeypatch, value):
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", value)
    with pytest.raises(ValueError):
        llm.requests_per_minute({"requests_per_minute": 10})
