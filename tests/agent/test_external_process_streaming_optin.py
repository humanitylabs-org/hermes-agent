"""External-process streaming is a provider capability, not a vendor exception."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_api_call import _should_stream
from providers.base import ProviderProfile


@pytest.mark.parametrize("base_url", ["process://cli", "https://proxy.invalid/v1", "acp://cli", "acp+tcp://localhost:9999"])
@pytest.mark.parametrize("optin", [False, True])
def test_external_process_streaming_requires_explicit_optin(monkeypatch, base_url, optin):
    profile = ProviderProfile(name="streaming-cli", auth_type="external_process", supports_streaming=optin)
    monkeypatch.setattr("providers.get_provider_profile", lambda _: profile)
    agent = SimpleNamespace(provider=profile.name, base_url=base_url, _has_stream_consumers=lambda: True)
    assert _should_stream(agent) is optin


@pytest.mark.parametrize("provider", ["copilot-acp", "unknown-acp"])
def test_no_flag_or_missing_profile_still_does_not_stream(monkeypatch, provider):
    profile = SimpleNamespace(auth_type="external_process") if provider == "copilot-acp" else None
    monkeypatch.setattr("providers.get_provider_profile", lambda _: profile)
    agent = SimpleNamespace(provider=provider, base_url="acp://cli", _has_stream_consumers=lambda: True)
    assert _should_stream(agent) is False


def test_optin_preserves_disable_and_mock_guards(monkeypatch):
    profile = ProviderProfile(name="streaming-cli", auth_type="external_process", supports_streaming=True)
    monkeypatch.setattr("providers.get_provider_profile", lambda _: profile)
    agent = SimpleNamespace(provider=profile.name, base_url="process://cli", _has_stream_consumers=lambda: False, client=Mock())
    assert _should_stream(agent) is False
    agent.client = object()
    assert _should_stream(agent) is True
    agent._disable_streaming = True
    assert _should_stream(agent) is False


def test_http_and_moa_behavior_unchanged(monkeypatch):
    monkeypatch.setattr("providers.get_provider_profile", lambda _: None)
    agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com", _has_stream_consumers=lambda: False, client=object())
    assert _should_stream(agent) is True
    agent.provider = "moa"
    assert _should_stream(agent) is False
