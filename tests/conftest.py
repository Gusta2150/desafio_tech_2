"""Configuração comum dos testes.

Roda antes de qualquer import de ``src``: os testes nunca leem o ``.env`` de
quem está executando e nunca chamam uma LLM real, mesmo que a pessoa tenha
``GROQ_API_KEY`` ou ``LLM_PROVIDER=groq`` configurados na máquina.
"""

import os

os.environ["TC_SKIP_DOTENV"] = "1"
os.environ["LLM_PROVIDER"] = "offline"
