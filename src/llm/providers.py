"""Provedores de LLM.

* ``openai``    — qualquer API compatível com OpenAI Chat Completions:
                  OpenAI (GPT), Groq (LLaMA 3, gratuito), Ollama local
                  (``http://localhost:11434/v1``), LM Studio, Together, Gemini etc.
* ``anthropic`` — API Messages da Anthropic (Claude).
* ``offline``   — gerador determinístico baseado em template. Não é uma LLM:
                  existe para a API, os testes e a demonstração funcionarem sem
                  chave/rede, e como *fallback* quando o provedor real falha.

Configuração por variáveis de ambiente (ver ``.env.example``)::

    LLM_PROVIDER=openai|anthropic|offline
    LLM_MODEL=llama-3.3-70b-versatile
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_API_KEY=...
    LLM_MAX_RETRIES=5      # novas tentativas em limite de taxa (429) e erros temporários

Limite de taxa: o plano gratuito do Groq limita tokens e requisições por minuto.
Quando a API responde 429 (ou 5xx, ou dá timeout), a chamada espera o tempo
indicado no cabeçalho ``Retry-After`` (ou um recuo exponencial) e tenta de novo.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_WAIT_S = 60.0

# Substituível nos testes para não esperar de verdade.
_sleep = time.sleep


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 900,
                 json_mode: bool = False, context: dict[str, Any] | None = None) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class LLMAPIError(LLMError):
    """Falha de comunicação com a API (rede, autenticação, limite de taxa esgotado).

    Separada de ``LLMError`` para distinguir "a LLM não respondeu" de "a LLM
    respondeu fora do formato" — só o segundo caso diz algo sobre a qualidade
    do prompt.
    """


def _max_retries() -> int:
    try:
        return max(0, int(os.getenv("LLM_MAX_RETRIES") or 5))
    except ValueError:
        return 5


def _retry_after(resp: httpx.Response | None, attempt: int) -> float:
    """Segundos de espera: ``Retry-After`` da API ou recuo exponencial com jitter."""
    if resp is not None:
        header = resp.headers.get("retry-after")
        if header:
            try:
                return min(MAX_WAIT_S, float(header) + 0.5)
            except ValueError:
                pass
    return min(MAX_WAIT_S, 2.0 ** attempt + random.uniform(0, 1))


def post_with_retry(url: str, *, json_body: dict[str, Any], headers: dict[str, str], timeout: float,
                    label: str) -> httpx.Response:
    """POST com novas tentativas em 429/5xx/timeout. Erros definitivos (ex.: 401) falham na hora."""
    retries = _max_retries()
    for attempt in range(retries + 1):
        resp: httpx.Response | None = None
        try:
            resp = httpx.post(url, json=json_body, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == retries:
                raise LLMAPIError(f"Falha ao chamar {label}: {exc}") from exc
            wait = _retry_after(None, attempt)
            logger.warning("%s: erro de rede (%s); nova tentativa em %.1fs", label, exc, wait)
            _sleep(wait)
            continue
        if resp.status_code < 400:
            return resp
        if resp.status_code in RETRY_STATUS and attempt < retries:
            wait = _retry_after(resp, attempt)
            logger.warning("%s: HTTP %d; nova tentativa %d/%d em %.1fs",
                           label, resp.status_code, attempt + 1, retries, wait)
            _sleep(wait)
            continue
        detail = resp.text[:300].replace("\n", " ")
        raise LLMAPIError(f"Falha ao chamar {label}: HTTP {resp.status_code} — {detail}")
    raise LLMAPIError(f"Falha ao chamar {label}: tentativas esgotadas")  # pragma: no cover


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"  # Ollama ignora a chave
        self.timeout = timeout

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        resp = post_with_retry(f"{self.base_url}/chat/completions", json_body=body, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {self.api_key}"}, label=self.base_url)
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMAPIError(f"Resposta inesperada de {self.base_url}: {resp.text[:200]}") from exc
        return LLMResponse(
            text=text, provider=self.name, model=self.model,
            latency_s=round(time.perf_counter() - start, 3), usage=data.get("usage", {}),
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None, base_url: str = "https://api.anthropic.com",
                 timeout: float = 60.0):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY/LLM_API_KEY não definida.")
        self.model, self.api_key, self.base_url, self.timeout = model, api_key, base_url.rstrip("/"), timeout

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        if json_mode:
            user += "\n\nResponda somente com o objeto JSON, sem texto antes ou depois."
        start = time.perf_counter()
        resp = post_with_retry(
            f"{self.base_url}/v1/messages", timeout=self.timeout, label="a API da Anthropic",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json_body={"model": self.model, "system": system, "max_tokens": max_tokens,
                       "temperature": temperature, "messages": [{"role": "user", "content": user}]},
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMAPIError(f"Resposta inesperada da API da Anthropic: {resp.text[:200]}") from exc
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_s=round(time.perf_counter() - start, 3), usage=data.get("usage", {}))


class OfflineTemplateProvider:
    """Gera a resposta no formato esperado a partir do contexto estruturado.

    Determinístico e sem rede. Usado em testes, na demo sem chave e como
    fallback. Não substitui uma LLM — serve de referência mínima na avaliação.
    """

    name = "offline"
    model = "template-v1"

    def generate(self, system, user, *, temperature=0.2, max_tokens=900, json_mode=False, context=None):
        start = time.perf_counter()
        if context is None:
            raise LLMError("O provedor offline precisa do contexto estruturado.")
        if "fatores_principais" in context:
            payload = self._case(context)
        else:
            payload = self._performance(context)
        text = json.dumps(payload, ensure_ascii=False) if json_mode else self._as_text(payload)
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_s=round(time.perf_counter() - start, 4))

    @staticmethod
    def _case(ctx: dict[str, Any]) -> dict[str, Any]:
        m = ctx["modelo"]
        alerta = ctx["classificacao_modelo"] != "sem alerta"
        fatores = [
            f"{f['nome']}: {f['valor']} (média benignos {f['media_benignos']}, média malignos "
            f"{f['media_malignos']}) — {f['direcao']}"
            for f in ctx["fatores_principais"][:3]
        ]
        return {
            "resumo": (
                f"O modelo estimou probabilidade de malignidade de {ctx['probabilidade_percentual']}%, "
                f"{'acima' if alerta else 'abaixo'} do limiar de decisão de {ctx['limiar_decisao']}. "
                f"Faixa de risco: {ctx['faixa_risco']}."
            ),
            "fatores_relevantes": fatores,
            "confiabilidade": (
                f"No conjunto de teste, o modelo ({m['nome']}, {m['origem']}) identificou "
                f"{m['recall_teste_percentual']}% dos casos malignos (recall), com "
                f"{m['falsos_negativos_teste']} falso(s) negativo(s) em {m['amostras_teste']} amostras."
            ),
            "recomendacao": (
                "Priorizar avaliação médica e considerar confirmação por exame de imagem e biópsia, "
                "conforme julgamento clínico." if alerta else
                "Sem alerta pelo limiar técnico; manter o acompanhamento clínico habitual e correlacionar "
                "com exame físico e imagem."
            ),
            "limitacoes": (
                "Estimativa estatística baseada em medidas morfológicas de núcleos celulares; "
                "não considera histórico clínico, imagem ou outros exames."
            ),
            "aviso": "Apoio à decisão: não confirma nem descarta câncer e não substitui a avaliação médica.",
        }

    @staticmethod
    def _performance(ctx: dict[str, Any]) -> dict[str, Any]:
        b, o = ctx["original"], ctx["otimizado"]
        delta_fn = b["falsos_negativos"] - o["falsos_negativos"]
        return {
            "resumo": (
                f"O modelo {ctx['modelo']} otimizado por algoritmo genético obteve recall de "
                f"{o['recall_percentual']}% contra {b['recall_percentual']}% do modelo original no teste."
            ),
            "insights": [
                f"Falsos negativos no teste: {b['falsos_negativos']} (original) → {o['falsos_negativos']} (otimizado).",
                f"Falsos positivos no teste: {b['falsos_positivos']} (original) → {o['falsos_positivos']} (otimizado).",
                f"ROC-AUC: {b['roc_auc']} (original) → {o['roc_auc']} (otimizado).",
            ],
            "impacto_clinico": (
                f"{'Menos' if delta_fn > 0 else 'Mesmo número de' if delta_fn == 0 else 'Mais'} casos malignos "
                "deixariam de ser sinalizados na triagem."
            ),
            "recomendacao": "Validar o modelo em dados externos antes de qualquer uso assistencial.",
            "aviso": "Métricas obtidas em base pública e amostra pequena; não substituem validação clínica.",
        }

    @staticmethod
    def _as_text(payload: dict[str, Any]) -> str:
        lines = []
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:\n" + "\n".join(f"- {v}" for v in value))
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


PRESETS = {
    "groq": ("openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "ollama": ("openai", "http://localhost:11434/v1", "llama3.1:8b", None),
    "openai": ("openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "https://api.anthropic.com", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
}


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Cria o provedor a partir do nome (ou de ``LLM_PROVIDER``).

    Aceita os atalhos ``groq``, ``ollama``, ``openai``, ``anthropic`` e ``offline``.
    """
    # ``or`` em vez do padrão do getenv: no .env as linhas "LLM_MODEL=" vazias
    # devem valer como "não definido", e não como string vazia.
    name = (name or os.getenv("LLM_PROVIDER") or "offline").lower()
    if name == "offline":
        return OfflineTemplateProvider()
    if name not in PRESETS:
        raise ValueError(f"Provedor desconhecido: {name}. Use offline, {', '.join(PRESETS)}.")
    kind, default_url, default_model, key_env = PRESETS[name]
    base_url = os.getenv("LLM_BASE_URL") or default_url
    model = model or os.getenv("LLM_MODEL") or default_model
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None)
    if key_env and not api_key:
        raise LLMError(f"Chave do provedor '{name}' não encontrada. Defina {key_env} (ou LLM_API_KEY) "
                       "no arquivo .env ou no ambiente.")
    if kind == "anthropic":
        return AnthropicProvider(model, api_key, base_url)
    provider = OpenAICompatibleProvider(model, base_url, api_key)
    provider.name = name
    return provider
