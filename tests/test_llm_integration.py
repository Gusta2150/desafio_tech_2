"""Integração com LLM real: .env, novas tentativas em limite de taxa e falhas visíveis na avaliação.

Nenhum teste aqui acessa a rede: ``httpx.post`` é substituído por respostas falsas.
"""

import json
import shutil
import sys

import httpx
import pytest

from src import config
from src.data import PROJECT_DIR
from src.llm import providers
from src.llm.providers import LLMAPIError, LLMError, LLMResponse, OpenAICompatibleProvider, get_provider

URL = "https://api.groq.com/openai/v1"


def _resp(status, body=None, headers=None):
    request = httpx.Request("POST", f"{URL}/chat/completions")
    if body is None:
        return httpx.Response(status, text="erro", headers=headers or {}, request=request)
    return httpx.Response(status, json=body, headers=headers or {}, request=request)


OK = {"choices": [{"message": {"content": '{"resumo": "ok"}'}}], "usage": {"total_tokens": 10}}


@pytest.fixture
def waits(monkeypatch):
    """Registra as esperas em vez de dormir de verdade."""
    recorded = []
    monkeypatch.setattr(providers, "_sleep", recorded.append)
    return recorded


def _fake_post(monkeypatch, sequence):
    calls = iter(sequence)

    def fake(url, **kw):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(providers.httpx, "post", fake)


# --------------------------------------------------------------------------- #
# Novas tentativas                                                             #
# --------------------------------------------------------------------------- #
def test_retries_on_429_using_retry_after(monkeypatch, waits):
    _fake_post(monkeypatch, [_resp(429, headers={"retry-after": "3"}), _resp(429), _resp(200, OK)])
    out = OpenAICompatibleProvider("llama", URL, "chave").generate("s", "u")
    assert out.text == '{"resumo": "ok"}'
    assert len(waits) == 2
    assert waits[0] == pytest.approx(3.5)        # Retry-After + folga
    assert 2.0 <= waits[1] <= 3.0                 # recuo exponencial (2^1 + jitter)


def test_retries_on_timeout(monkeypatch, waits):
    _fake_post(monkeypatch, [httpx.ReadTimeout("lento"), _resp(200, OK)])
    assert OpenAICompatibleProvider("llama", URL, "chave").generate("s", "u").text
    assert len(waits) == 1


def test_auth_error_fails_without_retry(monkeypatch, waits):
    _fake_post(monkeypatch, [_resp(401)])
    with pytest.raises(LLMAPIError, match="401"):
        OpenAICompatibleProvider("llama", URL, "errada").generate("s", "u")
    assert waits == []


def test_gives_up_after_max_retries(monkeypatch, waits):
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    _fake_post(monkeypatch, [_resp(429), _resp(429), _resp(429)])
    with pytest.raises(LLMAPIError, match="429"):
        OpenAICompatibleProvider("llama", URL, "chave").generate("s", "u")
    assert len(waits) == 2


def test_api_error_is_still_an_llm_error():
    # o código antigo (API, interpretador) captura LLMError e continua funcionando
    assert issubclass(LLMAPIError, LLMError)


# --------------------------------------------------------------------------- #
# Configuração                                                                 #
# --------------------------------------------------------------------------- #
def test_empty_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "chave")
    monkeypatch.setenv("LLM_MODEL", "")       # linhas vazias do .env.example
    monkeypatch.setenv("LLM_BASE_URL", "")
    p = get_provider("groq")
    assert p.model == "openai/gpt-oss-120b"
    assert p.base_url == URL


def test_reasoning_models_use_low_effort(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "chave")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    sent = []
    monkeypatch.setattr(providers.httpx, "post", lambda url, **kw: sent.append(kw["json"]) or _resp(200, OK))
    get_provider("groq").generate("s", "u")
    assert sent[0]["reasoning_effort"] == "low"
    get_provider("groq", model="outro-modelo").generate("s", "u")
    assert "reasoning_effort" not in sent[1]          # modelos comuns não recebem o parâmetro


def test_missing_key_fails_clearly(monkeypatch):
    for var in ("GROQ_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_provider("groq")


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert get_provider("ollama").model == "llama3.1:8b"


def test_load_env_reads_file_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TC_TESTE_A=do_arquivo\nTC_TESTE_B=do_arquivo\n", encoding="utf-8")
    monkeypatch.delenv("TC_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("TC_TESTE_A", raising=False)
    monkeypatch.setenv("TC_TESTE_B", "do_terminal")
    assert config.load_env(env) is True
    import os
    assert os.environ["TC_TESTE_A"] == "do_arquivo"
    assert os.environ["TC_TESTE_B"] == "do_terminal"
    monkeypatch.delenv("TC_TESTE_A")


def test_load_env_can_be_disabled(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TC_TESTE_C=x\n", encoding="utf-8")
    monkeypatch.setenv("TC_SKIP_DOTENV", "1")
    assert config.load_env(env) is False


# --------------------------------------------------------------------------- #
# Tipo de erro na interpretação e avaliação                                    #
# --------------------------------------------------------------------------- #
class ApiDown:
    name, model = "groq", "llama"

    def generate(self, system, user, **kw):
        raise LLMAPIError("HTTP 429 — limite de taxa")


class BadFormat:
    name, model = "groq", "llama"

    def generate(self, system, user, **kw):
        return LLMResponse("isto não é JSON", "groq", "llama", 0.1)


@pytest.fixture(scope="module")
def model_and_card():
    import joblib
    model = joblib.load(PROJECT_DIR / "models" / "best_model.joblib")
    card = json.loads((PROJECT_DIR / "models" / "model_card.json").read_text(encoding="utf-8"))
    return model, card


def test_interpretation_records_error_kind(model_and_card):
    from src.data import load_split
    from src.llm import DiagnosisInterpreter
    model, card = model_and_card
    sample = load_split().X_test.iloc[[0]]
    api = DiagnosisInterpreter(model, card, ApiDown(), fallback=False).explain_case(sample)
    fmt = DiagnosisInterpreter(model, card, BadFormat(), fallback=False).explain_case(sample)
    assert api.error_kind == "api" and fmt.error_kind == "formato"
    used_template = DiagnosisInterpreter(model, card, ApiDown()).explain_case(sample)
    assert used_template.used_fallback and used_template.error_kind == "api"


def test_evaluation_refuses_incomplete_results(tmp_path, monkeypatch):
    """Com falha de API, a avaliação não vai para reports/results (e o script sai com erro)."""
    import scripts.evaluate_llm as ev

    results = tmp_path / "results"
    results.mkdir()
    shutil.copy(PROJECT_DIR / "reports" / "results" / "experiments.json", results / "experiments.json")
    monkeypatch.setattr(ev, "RESULTS", results)
    monkeypatch.setattr(ev, "get_provider", lambda name=None, model=None: ApiDown())
    monkeypatch.setattr(sys, "argv", ["evaluate_llm", "--provider", "groq", "--cases", "2",
                                      "--strategies", "basico"])
    with pytest.raises(SystemExit) as exit_info:
        ev.main()
    assert exit_info.value.code == 1
    assert not (results / "llm_evaluation_groq.json").exists()
    saved = json.loads((results / "incompleto" / "llm_evaluation_groq.json").read_text(encoding="utf-8"))
    assert saved["falhas"]["api"] == 3   # 2 casos + relatório de desempenho
