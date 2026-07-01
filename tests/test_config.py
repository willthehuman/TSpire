from tspire.host.config import HostConfig


def test_default_ollama_model_is_cloud_model(monkeypatch):
    monkeypatch.delenv("TSPIRE_OLLAMA_MODEL", raising=False)
    cfg = HostConfig.load(path="missing-config.json")
    assert cfg.ollama_model == "gemma4:31b-cloud"
    assert cfg.input_backend == "mouse"
    assert cfg.ollama_think is False


def test_ollama_env_overrides(monkeypatch):
    monkeypatch.setenv("TSPIRE_VISION_MODE", "llm")
    monkeypatch.setenv("TSPIRE_OLLAMA_URL", "http://example.test:11434")
    monkeypatch.setenv("TSPIRE_OLLAMA_MODEL", "gemma4:31b-cloud")
    monkeypatch.setenv("TSPIRE_LLM_IMAGE_WIDTH", "768")
    monkeypatch.setenv("TSPIRE_OLLAMA_THINK", "true")

    cfg = HostConfig.load(path="missing-config.json")

    assert cfg.vision_mode == "llm"
    assert cfg.ollama_url == "http://example.test:11434"
    assert cfg.ollama_model == "gemma4:31b-cloud"
    assert cfg.llm_image_width == 768
    assert cfg.ollama_think is True


def test_easyocr_env_override(monkeypatch):
    monkeypatch.setenv("TSPIRE_USE_EASYOCR", "false")

    cfg = HostConfig.load(path="missing-config.json")

    assert cfg.use_easyocr is False


def test_input_backend_env_override(monkeypatch):
    monkeypatch.setenv("TSPIRE_INPUT_BACKEND", "gamepad")
    cfg = HostConfig.load(path="missing-config.json")
    assert cfg.input_backend == "gamepad"


def test_focus_before_capture_env_override(monkeypatch):
    monkeypatch.setenv("TSPIRE_FOCUS_BEFORE_CAPTURE", "false")
    cfg = HostConfig.load(path="missing-config.json")
    assert cfg.focus_before_capture is False
