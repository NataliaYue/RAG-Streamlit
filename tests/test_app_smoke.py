"""Tests de humo de la interfaz con streamlit.testing (no requieren navegador ni red)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "frontend" / "app.py"


@pytest.fixture(autouse=True)
def sin_dotenv(monkeypatch):
    """Evita que un .env real del desarrollador altere el resultado de los tests."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


def test_sin_api_key_muestra_error_y_se_detiene(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)
    assert len(at.metric) == 0  # st.stop() impide construir el resto de la UI


def test_con_api_key_renderiza_la_interfaz_inicial(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")

    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert len(at.error) == 0
    assert [m.label for m in at.metric] == ["Documentos", "Fragmentos"]
    assert [m.value for m in at.metric] == ["0", "0"]
    assert any("Sube un documento" in i.value for i in at.info)


def test_consultar_sin_indexar_muestra_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")

    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.chat_input[0].set_value("¿Qué significa la luz roja?").run()

    assert not at.exception
    assert any("Indexa al menos un documento" in e.value for e in at.error)
    assert len(at.chat_message) == 0
