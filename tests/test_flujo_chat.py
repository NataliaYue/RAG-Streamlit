"""Flujo completo del chat: pregunta -> reformulación -> búsqueda -> respuesta -> fuentes.

El LLM se sustituye por cadenas falsas; la búsqueda usa un índice real con embeddings falsos.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend import document_processor as dp
from backend import llm_chain as lc
from backend.document_processor import IndiceDocumental

APP = Path(__file__).resolve().parent.parent / "frontend" / "app.py"

PAGINAS = [
    "El error E-42 indica sobrecalentamiento del motor",
    "Instrucciones de limpieza del filtro cada semana",
    "Garantia y condiciones de devolucion del producto",
]


class CadenaStreamFalsa:
    """Sustituye a la cadena RAG: emite trozos de texto y registra lo que recibe."""

    def __init__(self, trozos=None, error=None):
        self.trozos = trozos or []
        self.error = error
        self.entradas = []

    def stream(self, datos):
        self.entradas.append(datos)
        if self.error:
            raise self.error
        yield from self.trozos


class ReformuladorFalso:
    def __init__(self, resultado):
        self.resultado = resultado
        self.llamadas = []

    def invoke(self, datos):
        self.llamadas.append(datos)
        return self.resultado


@pytest.fixture(autouse=True)
def entorno(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")


@pytest.fixture
def indice(archivo, crear_pdf):
    indice = IndiceDocumental()
    indice.agregar_archivos([archivo("manual.pdf", crear_pdf(PAGINAS))])
    return indice


def abrir_app(indice: IndiceDocumental) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["indice"] = indice
    return at.run()


def preguntar(at: AppTest, texto: str) -> AppTest:
    return at.chat_input[0].set_value(texto).run()


# --- Respuesta normal -------------------------------------------------------------------------
def test_respuesta_con_fuentes_y_fragmentos(monkeypatch, indice):
    cadena = CadenaStreamFalsa(["El error E-42 ", "indica sobrecalentamiento."])
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)

    at = preguntar(abrir_app(indice), "¿Qué significa E-42?")

    assert not at.exception
    mensajes = at.session_state["messages"]
    assert [m["role"] for m in mensajes] == ["user", "assistant"]
    assert mensajes[1]["content"] == "El error E-42 indica sobrecalentamiento."
    assert "manual.pdf" in mensajes[1]["fuentes"]
    assert mensajes[1]["fragmentos"]
    assert any("manual.pdf" in m.value for m in at.markdown)

    # El LLM recibe la pregunta original y el contexto recuperado con su procedencia
    entrada = cadena.entradas[0]
    assert entrada["input"] == "¿Qué significa E-42?"
    assert "E-42" in entrada["context"]
    assert 'fuente="manual.pdf"' in entrada["context"]


def test_sin_respuesta_no_muestra_fuentes(monkeypatch, indice):
    cadena = CadenaStreamFalsa([lc.SIN_RESPUESTA])
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)

    at = preguntar(abrir_app(indice), "¿Cuál es la capital de Francia?")

    assert not at.exception
    respuesta = at.session_state["messages"][1]
    assert respuesta["content"] == lc.SIN_RESPUESTA
    assert "fuentes" not in respuesta
    assert "fragmentos" not in respuesta


# --- Errores -------------------------------------------------------------------------------------
def test_error_del_modelo_se_muestra_y_no_deja_pregunta_huerfana(monkeypatch, indice):
    cadena = CadenaStreamFalsa(error=RuntimeError("rate limit"))
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)

    at = preguntar(abrir_app(indice), "¿Qué significa E-42?")

    assert not at.exception
    assert any("No se pudo generar la respuesta" in e.value for e in at.error)
    assert at.session_state["messages"] == []


# --- Reformulación ------------------------------------------------------------------------------
def test_segunda_pregunta_se_reformula_y_se_busca_con_la_reformulada(
    monkeypatch, indice
):
    cadena = CadenaStreamFalsa(["Respuesta"])
    reformulador = ReformuladorFalso("¿Cómo se soluciona el error E-42?")
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)
    monkeypatch.setattr(lc, "inicializar_reformulador", lambda: reformulador)

    busquedas = []
    buscar_original = indice.buscar

    def buscar_espia(pregunta, *args, **kwargs):
        busquedas.append(pregunta)
        return buscar_original(pregunta, *args, **kwargs)

    monkeypatch.setattr(indice, "buscar", buscar_espia)

    at = abrir_app(indice)
    preguntar(at, "¿Qué es E-42?")
    assert reformulador.llamadas == []  # la primera pregunta no tiene historial

    preguntar(at, "¿y cómo se arregla?")

    assert not at.exception
    assert len(reformulador.llamadas) == 1
    assert "¿Qué es E-42?" in reformulador.llamadas[0]["chat_history"]
    assert busquedas == ["¿Qué es E-42?", "¿Cómo se soluciona el error E-42?"]
    assert (
        cadena.entradas[1]["input"] == "¿y cómo se arregla?"
    )  # se responde a la original
    assert any("Buscando en el manual" in c.value for c in at.caption)


# --- Sin documentos ------------------------------------------------------------------------------
def test_consultar_sin_indexar_no_invoca_al_modelo(monkeypatch):
    cadena = CadenaStreamFalsa(["no debería llamarse"])
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)

    at = preguntar(abrir_app(IndiceDocumental()), "¿Hola?")

    assert any("Indexa al menos un documento" in e.value for e in at.error)
    assert cadena.entradas == []


# --- Persistencia desde la app -----------------------------------------------------------------
def test_la_app_carga_el_indice_guardado_al_arrancar(monkeypatch, tmp_path, indice):
    destino = tmp_path / "guardado"
    indice.guardar(destino)
    monkeypatch.setattr(dp, "INDEX_DIR", destino)

    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert [m.value for m in at.metric] == ["1", str(indice.n_fragmentos)]


def test_indice_guardado_corrupto_avisa_y_arranca_vacio(monkeypatch, tmp_path):
    destino = tmp_path / "corrupto"
    destino.mkdir()
    (destino / "indice.json").write_text("esto no es json", encoding="utf-8")
    monkeypatch.setattr(dp, "INDEX_DIR", destino)

    at = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not at.exception
    assert any("No se pudo cargar el índice guardado" in w.value for w in at.warning)
    assert [m.value for m in at.metric] == ["0", "0"]


def test_purgar_borra_el_indice_de_memoria_y_de_disco(monkeypatch, tmp_path, indice):
    destino = tmp_path / "guardado"
    indice.guardar(destino)
    monkeypatch.setattr(dp, "INDEX_DIR", destino)

    at = AppTest.from_file(str(APP), default_timeout=30).run()
    purgar = next(b for b in at.button if b.label == "Purgar base de datos")
    purgar.click().run()

    assert not at.exception
    assert [m.value for m in at.metric] == ["0", "0"]
    assert not (destino / "indice.json").exists()
