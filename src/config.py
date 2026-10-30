"""Carrega as variáveis do arquivo ``.env`` da raiz do projeto.

Os scripts, a API e o notebook chamam ``load_env()`` no início. Variáveis que
já existem no ambiente (por exemplo, exportadas no terminal ou definidas pelo
Docker/Kubernetes) têm prioridade e não são sobrescritas pelo ``.env``.

``TC_SKIP_DOTENV=1`` desliga a leitura. Os testes usam isso para não depender
do ``.env`` de quem está rodando (e para nunca chamarem uma LLM real).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path | None = None) -> bool:
    """Lê o ``.env`` se ele existir. Devolve ``True`` quando um arquivo foi lido."""
    if os.getenv("TC_SKIP_DOTENV"):
        return False
    path = path or ENV_FILE
    if not path.exists():
        return False
    load_dotenv(path, override=False)
    return True
