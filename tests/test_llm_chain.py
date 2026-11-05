import pytest
from langchain_core.prompts import ChatPromptTemplate

from backend import llm_chain as lc


class CadenaFalsa:
    def __init__(self, resultado=None, error=None):
        self.resultado = resultado
        self.error = error
        self.llamadas = []

    def invoke(self, datos):
        self.llamadas.append(datos)
        if self.error:
            raise self.error
        return self.resultado


@pytest.fixture(autouse=True)
def limpiar_caches():
    # Se guardan las funciones originales: al terminar, un test puede haberlas
    # sustituido con monkeypatch y la sustituta no tiene cache_clear().
    cadena_rag = lc.inicializar_cadena_rag
    reformulador = lc.inicializar_reformulador
    cadena_rag.cache_clear()
    reformulador.cache_clear()
    yield
    cadena_rag.cache_clear()
    reformulador.cache_clear()


# --- formatear_historial ---------------------------------------------------------
class TestHistorial:
    def test_asigna_roles(self):
        texto = lc.formatear_historial(
            [
                {"role": "user", "content": "hola"},
                {"role": "assistant", "content": "qué tal"},
            ]
        )
        assert texto == "Usuario: hola\nAsistente: qué tal"

    def test_limita_numero_de_mensajes(self):
        mensajes = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        lineas = lc.formatear_historial(mensajes, max_mensajes=4).splitlines()
        assert len(lineas) == 4
        assert lineas[0].endswith("msg6")
        assert lineas[-1].endswith("msg9")

    def test_trunca_mensajes_largos(self):
        texto = lc.formatear_historial(
            [{"role": "assistant", "content": "x" * 1000}], max_chars=300
        )
        assert len(texto) < 330
        assert texto.endswith("…")

    def test_ignora_fuentes_guardadas_aparte(self):
        # Las fuentes viven en msg["fuentes"], no en "content": no deben colarse.
        texto = lc.formatear_historial(
            [
                {
                    "role": "assistant",
                    "content": "respuesta",
                    "fuentes": "**Fuentes:** a.pdf",
                }
            ]
        )
        assert "Fuentes" not in texto

    def test_historial_vacio(self):
        assert lc.formatear_historial([]) == ""


# --- reformular_pregunta ------------------------------------------------------------
class TestReformulacion:
    def test_sin_historial_no_llama_al_modelo(self, monkeypatch):
        cadena = CadenaFalsa(resultado="no debería usarse")
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: cadena)

        assert lc.reformular_pregunta("", "¿Cómo reinicio?") == "¿Cómo reinicio?"
        assert cadena.llamadas == []

    def test_usa_la_pregunta_reformulada_y_hace_strip(self, monkeypatch):
        cadena = CadenaFalsa(resultado="  ¿Cómo reinicio el router?\n")
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: cadena)

        resultado = lc.reformular_pregunta(
            "Usuario: hablemos del router", "¿y cómo lo reinicio?"
        )

        assert resultado == "¿Cómo reinicio el router?"
        assert cadena.llamadas == [
            {
                "chat_history": "Usuario: hablemos del router",
                "input": "¿y cómo lo reinicio?",
                "instruccion_idioma": lc.INSTRUCCION_MANTENER_IDIOMA,
            }
        ]

    def test_resultado_vacio_usa_la_original(self, monkeypatch):
        monkeypatch.setattr(
            lc, "inicializar_reformulador", lambda: CadenaFalsa(resultado="   ")
        )
        assert lc.reformular_pregunta("historial", "pregunta") == "pregunta"

    def test_si_el_modelo_falla_usa_la_original(self, monkeypatch):
        cadena = CadenaFalsa(error=RuntimeError("rate limit"))
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: cadena)
        assert lc.reformular_pregunta("historial", "pregunta") == "pregunta"


# --- Prompts ------------------------------------------------------------------------------
class TestPrompts:
    def test_prompt_rag_se_renderiza_con_todas_las_variables(self):
        prompt = ChatPromptTemplate.from_messages(
            [("system", lc._SISTEMA), ("human", lc._HUMANO)]
        )
        mensajes = prompt.format_messages(
            chat_history="hist", context="ctx", input="¿qué?"
        )

        assert lc.SIN_RESPUESTA in mensajes[0].content
        assert "<contexto>\nctx\n</contexto>" in mensajes[1].content
        assert "<historial>\nhist\n</historial>" in mensajes[1].content
        assert "¿qué?" in mensajes[1].content

    def test_prompt_del_reformulador_se_renderiza(self):
        texto = ChatPromptTemplate.from_template(lc._PROMPT_REFORMULADOR).format(
            chat_history="hist",
            input="pregunta",
            instruccion_idioma="Escribe en inglés.",
        )
        assert "hist" in texto and "pregunta" in texto and "Escribe en inglés." in texto

    @pytest.mark.parametrize(
        "codigo, nombre", [("en", "inglés"), ("es", "español"), ("de", "alemán")]
    )
    def test_instruccion_de_idioma_nombra_el_destino(self, codigo, nombre):
        assert nombre in lc.instruccion_idioma(codigo)

    def test_sin_destino_se_mantiene_el_idioma(self):
        assert lc.instruccion_idioma(None) == lc.INSTRUCCION_MANTENER_IDIOMA


# --- API key y construcción -------------------------------------------------------------
class TestApiKey:
    def test_hay_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "abc")
        assert lc.hay_api_key() is True
        monkeypatch.delenv("GROQ_API_KEY")
        assert lc.hay_api_key() is False

    def test_sin_clave_lanza_error_claro(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            lc.inicializar_cadena_rag()

    def test_cadenas_se_cachean(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")
        assert lc.inicializar_cadena_rag() is lc.inicializar_cadena_rag()
        assert lc.inicializar_reformulador() is lc.inicializar_reformulador()
