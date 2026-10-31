"""Rotas usadas pela aplicação web (só o provedor offline: nenhuma LLM real é chamada)."""

import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app
from src.experiments import RESULTS_DIR

PRIVATE = {"raw", "prompt_system", "prompt_user", "json_mode", "usage"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def samples(client):
    return client.get("/samples").json()


@pytest.fixture(scope="module")
def sample(samples):
    return samples["amostras"][0]["medidas"]


# --------------------------------------------------------------------------- #
# /samples                                                                     #
# --------------------------------------------------------------------------- #
def test_samples_cover_the_test_set(client, samples):
    features = client.get("/model").json()["features"]
    assert samples["total"] == len(samples["amostras"]) == 114
    row = samples["amostras"][0]
    assert set(row) >= {"indice", "diagnostico_real", "probabilidade", "alerta", "acertou", "medidas"}
    assert list(row["medidas"]) == features
    assert "aviso" in samples


def test_samples_put_errors_then_near_threshold_first(samples):
    rows = samples["amostras"]
    n_err = sum(not r["acertou"] for r in rows)
    assert n_err > 0
    assert all(not r["acertou"] for r in rows[:n_err])
    hits = [r["distancia_limiar"] for r in rows[n_err:]]
    assert hits == sorted(hits)


@pytest.mark.parametrize("filtro,check", [
    ("erros", lambda r: not r["acertou"]),
    ("malignos", lambda r: r["diagnostico_real"] == "M"),
    ("benignos", lambda r: r["diagnostico_real"] == "B"),
    ("limiar", lambda r: r["distancia_limiar"] <= main.NEAR_THRESHOLD),
])
def test_samples_filters(client, filtro, check):
    body = client.get("/samples", params={"filtro": filtro}).json()
    assert body["filtro"] == filtro and body["total"] > 0
    assert all(check(r) for r in body["amostras"])


def test_samples_invalid_filter(client):
    assert client.get("/samples", params={"filtro": "todos"}).status_code == 422


# --------------------------------------------------------------------------- #
# Contrato das rotas antigas: nada de raw nem prompts                          #
# --------------------------------------------------------------------------- #
def test_interpret_ask_and_report_hide_internal_fields(client, sample):
    bodies = [
        client.post("/interpret", json={"amostra": sample, "provider": "offline"}).json(),
        client.post("/ask", json={"amostra": sample, "pergunta": "Qual fator pesou?", "provider": "offline"}).json(),
        client.get("/report/performance", params={"provider": "offline"}).json(),
    ]
    for body in bodies:
        assert not PRIVATE & set(body), PRIVATE & set(body)
        assert body["text"]


# --------------------------------------------------------------------------- #
# /llm/inspect e /llm/compare                                                  #
# --------------------------------------------------------------------------- #
def test_inspect_returns_everything(client, sample):
    r = client.post("/llm/inspect", json={"amostra": sample, "estrategia": "estruturado", "provider": "offline"})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt_system"] and body["prompt_user"] and body["json_mode"] is True
    assert body["resposta_bruta"] and body["estruturado"]["resumo"] and body["texto"]
    assert body["contexto"]["fatores_principais"]
    assert {"formato", "fidelidade", "cobertura", "coerencia", "seguranca", "legibilidade", "score_geral",
            "numeros_inventados", "violacoes"} <= set(body["notas"])
    assert body["juiz"] is None and body["juiz_status"] == "nao_pedido"
    assert body["provider"] == "offline" and body["used_fallback"] is False and body["error"] is None
    assert "usage" in body and "latencia_s" in body and body["prompt_version"]


def test_inspect_judge_is_unavailable_offline(client, sample):
    body = client.post("/llm/inspect", json={"amostra": sample, "provider": "offline", "juiz": True}).json()
    assert body["juiz"] is None and body["juiz_status"] == "indisponivel_offline"


def test_inspect_rejects_unknown_provider(client, sample):
    assert client.post("/llm/inspect", json={"amostra": sample, "provider": "nao-existe"}).status_code == 400


def test_compare_runs_the_three_strategies(client, sample):
    body = client.post("/llm/compare", json={"amostra": sample, "provider": "offline"}).json()
    assert [b["estrategia"] for b in body] == ["basico", "estruturado", "few_shot"]
    assert len({b["prompt_user"] for b in body}) == 3
    assert all(b["notas"]["score_geral"] > 0 for b in body)


# --------------------------------------------------------------------------- #
# Avaliações salvas                                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    for f in RESULTS_DIR.glob("llm_evaluation_*.json"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "incompleto").mkdir()
    (tmp_path / "incompleto" / "llm_evaluation_parcial.json").write_text(json.dumps({"falhas": {"api": 3}}))
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_evaluations_list_ignores_incomplete(client, results_dir):
    body = client.get("/llm/evaluations").json()
    providers = {e["provider"] for e in body}
    assert "offline" in providers and "parcial" not in providers
    offline = next(e for e in body if e["provider"] == "offline")
    assert offline["cases"] > 0 and "total_falhas" in offline


def test_evaluation_detail(client, results_dir):
    body = client.get("/llm/evaluations/offline").json()
    assert set(body["resumo"]) == {"basico", "estruturado", "few_shot"}
    assert body["linhas"] and body["exemplos"] and "falhas" in body


def test_evaluation_unknown_provider(client, results_dir):
    assert client.get("/llm/evaluations/nao-existe").status_code == 404
    assert client.get("/llm/evaluations/Offline..").status_code == 404


# --------------------------------------------------------------------------- #
# Algoritmo genético                                                           #
# --------------------------------------------------------------------------- #
def test_experiments_history(client):
    body = client.get("/experiments/history").json()
    assert body["selecionado"]["model"] in body["modelos"]
    for model in body["modelos"].values():
        assert model["experimentos"]
        for exp in model["experimentos"].values():
            hist = exp["historico"]
            assert hist and {"geracao", "melhor", "media", "diversidade", "avaliacoes"} <= set(hist[0])
            assert hist[-1]["avaliacoes"] == exp["avaliacoes"]
        rs = model["busca_aleatoria"]
        assert len(rs["curva"]) == rs["orcamento"]


def test_optimize_job_exposes_live_history(client):
    job_id = client.post("/optimize", json={"modelo": "logistic_regression", "population_size": 4,
                                            "generations": 2}).json()["job_id"]
    for _ in range(120):
        status = client.get(f"/optimize/{job_id}").json()
        if status["status"] != "executando":
            break
        time.sleep(0.5)
    assert status["status"] == "concluido"
    assert len(status["historico"]) == len(status["resultado"]["history"])
    assert status["historico"][-1]["melhor"] == pytest.approx(status["resultado"]["history"][-1]["melhor"])
