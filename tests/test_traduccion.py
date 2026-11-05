"""Idioma del índice, traducción de la consulta y tokenización de BM25."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from backend import document_processor as dp
from backend import limpieza as L
from backend import llm_chain as lc
from backend.document_processor import IndiceDocumental
from evaluation.evaluar import main, traducir_casos

APP = Path(__file__).resolve().parent.parent / "frontend" / "app.py"

EN = (
    "Before you start the procedure make sure that your computer is turned off and that "
    "you have read all of the safety information in this guide because it is important"
)
ES = (
    "Antes de empezar el procedimiento asegurese de que el equipo esta apagado y de que "
    "ha leido toda la informacion de seguridad de esta guia porque es muy importante para usted"
)


def frag(texto: str, idioma: str | None) -> Document:
    meta = {"source": "m.pdf", "page": 1}
    if idioma:
        meta["idioma"] = idioma
    return Document(page_content=texto, metadata=meta)


def indice_con(idiomas: list[str | None]) -> IndiceDocumental:
    indice = IndiceDocumental()
    indice.fragmentos = [frag("x", i) for i in idiomas]
    return indice


# --- Tokenizador de BM25 ---------------------------------------------------------------------
class TestTokenizar:
    def test_minusculas_y_sin_puntuacion_pegada(self):
        assert L.tokenizar_bm25("Keyboard, Screen. Wifi?") == ["keyboard", "screen", "wifi"]

    def test_quita_palabras_vacias_de_varios_idiomas(self):
        tokens = L.tokenizar_bm25("How do I clean the screen de la pantalla und der")
        assert "the" not in tokens and "de" not in tokens and "der" not in tokens
        assert "clean" in tokens and "screen" in tokens and "pantalla" in tokens

    @pytest.mark.parametrize("codigo", ["E-42", "M.2", "nano-SIM", "0183"])
    def test_conserva_codigos_y_nombres_compuestos(self, codigo):
        assert codigo.lower() in L.tokenizar_bm25(f"see {codigo}.")

    def test_consulta_y_documento_coinciden_aunque_cambien_mayusculas(self):
        assert L.tokenizar_bm25("Fan error") == L.tokenizar_bm25("FAN ERROR")

    def test_descarta_letras_sueltas_pero_no_numeros(self):
        assert L.tokenizar_bm25("I a 5 x") == ["5"]

    @pytest.mark.parametrize(
        "variante, normal",
        [
            ("nano‑SIM", "nano-SIM"),  # guion no separable: lo que emite el traductor
            ("Wi‑Fi", "Wi-Fi"),
            ("nano‐SIM", "nano-SIM"),  # guion Unicode
            ("ﬁngerprint reader", "fingerprint reader"),  # ligadura "ﬁ" de los PDF
            ("key­board", "keyboard"),  # guion blando de una ruptura de línea
            ("ac power adapter", "ac power adapter"),  # espacios duros
        ],
    )
    def test_variantes_tipograficas_coinciden_con_la_forma_normal(self, variante, normal):
        assert L.tokenizar_bm25(variante) == L.tokenizar_bm25(normal)

    def test_el_guion_no_separable_mantiene_la_palabra_entera(self):
        assert L.tokenizar_bm25("nano‑SIM") == ["nano-sim"]

    def test_raya_entre_palabras_las_separa(self):
        assert L.tokenizar_bm25("screen—keyboard") == ["screen", "keyboard"]


# --- Idioma por fragmento y del índice -------------------------------------------------------------
class TestIdiomaIndice:
    def test_marcar_idiomas_guarda_el_idioma_detectado(self):
        frs = [Document(page_content=EN, metadata={"source": "m.pdf", "page": 1}),
               Document(page_content="muy corto", metadata={"source": "m.pdf", "page": 2})]
        L.marcar_idiomas(frs)
        assert frs[0].metadata["idioma"] == "en"
        assert "idioma" not in frs[1].metadata  # no concluyente: no se inventa

    def test_idioma_dominante(self):
        assert indice_con(["en"] * 30).idioma_principal == "en"

    def test_dominante_con_algo_de_ruido(self):
        assert indice_con(["en"] * 27 + ["de"] * 3).idioma_principal == "en"

    def test_sin_dominante_claro_no_hay_idioma(self):
        assert indice_con(["en"] * 15 + ["es"] * 15).idioma_principal is None

    def test_pocos_fragmentos_no_hay_idioma(self):
        assert indice_con(["en"] * 5).idioma_principal is None

    def test_fragmentos_sin_idioma_no_cuentan(self):
        assert indice_con(["en"] * 20 + [None] * 50).idioma_principal == "en"

    def test_indice_vacio(self):
        assert IndiceDocumental().idioma_principal is None

    def test_el_idioma_sobrevive_a_guardar_y_cargar(self, archivo, crear_pdf, tmp_path):
        paginas = [EN.replace("procedure", f"procedure{i}x") for i in range(25)]
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("m.pdf", crear_pdf(paginas))])
        assert indice.idioma_principal == "en"

        indice.guardar(tmp_path)
        assert IndiceDocumental.cargar(tmp_path).idioma_principal == "en"


# --- Reformulador con idioma de destino ---------------------------------------------------------------
class CadenaFalsa:
    def __init__(self, resultado="x"):
        self.resultado = resultado
        self.llamadas = []

    def invoke(self, datos):
        self.llamadas.append(datos)
        return self.resultado


class TestReformularConIdioma:
    def test_sin_historial_pero_con_destino_si_llama_al_modelo(self, monkeypatch):
        cadena = CadenaFalsa("How do I reset the router?")
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: cadena)

        resultado = lc.reformular_pregunta("", "¿Cómo reinicio el router?", idioma_destino="en")

        assert resultado == "How do I reset the router?"
        datos = cadena.llamadas[0]
        assert datos["chat_history"] == "(sin historial)"
        assert "inglés" in datos["instruccion_idioma"]

    def test_sin_historial_ni_destino_no_llama(self, monkeypatch):
        cadena = CadenaFalsa()
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: cadena)
        assert lc.reformular_pregunta("", "hola") == "hola"
        assert cadena.llamadas == []

    def test_si_falla_la_traduccion_se_busca_con_la_original(self, monkeypatch):
        class Rota:
            def invoke(self, datos):
                raise RuntimeError("rate limit")

        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: Rota())
        assert lc.reformular_pregunta("", "pregunta", idioma_destino="en") == "pregunta"


class TestReformularConsulta:
    """`reformular_consulta` avisa de si el modelo falló, para poder reaccionar."""

    def test_sin_llamada_es_ok(self, monkeypatch):
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: CadenaFalsa())
        assert lc.reformular_consulta("", "hola") == ("hola", True)

    def test_exito(self, monkeypatch):
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: CadenaFalsa(" Hello \n"))
        assert lc.reformular_consulta("", "hola", idioma_destino="en") == ("Hello", True)

    def test_error_del_modelo_no_es_ok(self, monkeypatch):
        class Rota:
            def invoke(self, datos):
                raise RuntimeError("model_not_found")

        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: Rota())
        assert lc.reformular_consulta("", "hola", idioma_destino="en") == ("hola", False)

    def test_respuesta_vacia_no_es_ok(self, monkeypatch):
        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: CadenaFalsa("   "))
        assert lc.reformular_consulta("", "hola", idioma_destino="en") == ("hola", False)

    def test_el_fallo_se_registra_en_una_linea_con_el_modelo(self, monkeypatch, caplog):
        class Rota:
            def invoke(self, datos):
                raise RuntimeError("model_not_found")

        monkeypatch.setattr(lc, "inicializar_reformulador", lambda: Rota())
        with caplog.at_level("WARNING", logger=lc.logger.name):
            lc.reformular_consulta("", "hola", idioma_destino="en")

        avisos = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(avisos) == 1
        assert avisos[0].exc_info is None  # sin traceback: se repetiría en cada pregunta
        assert lc.MODELO_REFORMULADOR in avisos[0].getMessage()
        assert "RuntimeError" in avisos[0].getMessage()

    def test_el_modelo_por_defecto_no_es_uno_fijo_que_tu_cuenta_pueda_no_tener(self):
        # Sin GROQ_REFORMULATOR_MODEL, el reformulador usa el modelo que responde.
        if not __import__("os").getenv("GROQ_REFORMULATOR_MODEL"):
            assert lc.MODELO_REFORMULADOR == lc.MODELO_RESPUESTA


# --- La app traduce solo cuando hay un idioma claro --------------------------------------------------
@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")


class CadenaStream:
    def __init__(self):
        self.entradas = []

    def stream(self, datos):
        self.entradas.append(datos)
        yield "respuesta"


def abrir(indice):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["indice"] = indice
    return at.run()


def test_la_app_busca_con_la_pregunta_traducida(entorno, monkeypatch, archivo, crear_pdf):
    paginas = [EN.replace("procedure", f"procedure{i}x") for i in range(25)]
    indice = IndiceDocumental()
    indice.agregar_archivos([archivo("m.pdf", crear_pdf(paginas))])
    assert indice.idioma_principal == "en"

    reformulador = CadenaFalsa("How do I reset the router?")
    cadena = CadenaStream()
    monkeypatch.setattr(lc, "inicializar_reformulador", lambda: reformulador)
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: cadena)
    busquedas = []
    original = indice.buscar
    monkeypatch.setattr(
        indice, "buscar", lambda q, *a, **k: busquedas.append(q) or original(q, *a, **k)
    )

    at = abrir(indice)
    at.chat_input[0].set_value("¿Cómo reinicio el router?").run()

    assert not at.exception
    assert busquedas == ["How do I reset the router?"]  # se busca con la traducción
    assert cadena.entradas[0]["input"] == "¿Cómo reinicio el router?"  # se responde a la original
    assert any("How do I reset the router?" in c.value for c in at.caption)


def indice_en(archivo, crear_pdf) -> IndiceDocumental:
    paginas = [EN.replace("procedure", f"procedure{i}x") for i in range(25)]
    indice = IndiceDocumental()
    indice.agregar_archivos([archivo("m.pdf", crear_pdf(paginas))])
    assert indice.idioma_principal == "en"
    return indice


def espiar_pesos(indice, monkeypatch) -> list:
    """Registra el `peso_lexico` con el que la app llama a buscar."""
    pesos = []
    original = indice.buscar
    monkeypatch.setattr(
        indice,
        "buscar",
        lambda q, *a, **k: pesos.append(k.get("peso_lexico")) or original(q, *a, **k),
    )
    return pesos


def test_si_la_traduccion_falla_avisa_y_baja_el_peso_de_bm25(
    entorno, monkeypatch, archivo, crear_pdf
):
    # El caso real: el modelo del reformulador no existe y todas las llamadas fallan.
    class Rota:
        def invoke(self, datos):
            raise RuntimeError("model_not_found")

    indice = indice_en(archivo, crear_pdf)
    monkeypatch.setattr(lc, "inicializar_reformulador", lambda: Rota())
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: CadenaStream())
    pesos = espiar_pesos(indice, monkeypatch)

    at = abrir(indice)
    at.chat_input[0].set_value("¿Cómo reinicio el router?").run()

    assert not at.exception  # la app sigue funcionando
    assert pesos == [dp.PESO_LEXICO_SIN_TRADUCCION]
    assert any("No se pudo traducir" in c.value for c in at.caption)


def test_si_la_traduccion_funciona_se_usa_el_peso_normal_de_bm25(
    entorno, monkeypatch, archivo, crear_pdf
):
    indice = indice_en(archivo, crear_pdf)
    monkeypatch.setattr(lc, "inicializar_reformulador", lambda: CadenaFalsa("How do I reset it?"))
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: CadenaStream())
    pesos = espiar_pesos(indice, monkeypatch)

    at = abrir(indice)
    at.chat_input[0].set_value("¿Cómo lo reinicio?").run()

    assert not at.exception
    assert pesos == [None]  # None = el peso por defecto (PESO_LEXICO)
    assert not any("No se pudo traducir" in c.value for c in at.caption)


def test_la_app_no_traduce_si_no_hay_idioma_claro(entorno, monkeypatch, archivo, crear_pdf):
    indice = IndiceDocumental()
    indice.agregar_archivos([archivo("m.pdf", crear_pdf(["Manual corto con poco texto"]))])
    assert indice.idioma_principal is None

    reformulador = CadenaFalsa("no debería usarse")
    monkeypatch.setattr(lc, "inicializar_reformulador", lambda: reformulador)
    monkeypatch.setattr(lc, "inicializar_cadena_rag", lambda: CadenaStream())

    at = abrir(indice)
    at.chat_input[0].set_value("¿Qué dice el manual?").run()

    assert not at.exception
    assert reformulador.llamadas == []


# --- Evaluación con --traducir ----------------------------------------------------------------------------
class TestEvaluarTraducir:
    @pytest.fixture
    def archivos(self, tmp_path, crear_pdf):
        carpeta = tmp_path / "pdfs"
        carpeta.mkdir()
        (carpeta / "manual.pdf").write_bytes(crear_pdf(["El error E-42 indica sobrecalentamiento"]))
        preguntas = tmp_path / "p.json"
        preguntas.write_text(
            json.dumps([{"pregunta": "¿Qué es E-42?", "fuente": "manual.pdf", "paginas": [1]}]),
            encoding="utf-8",
        )
        return carpeta, preguntas

    def test_traducir_casos_usa_el_reformulador_sin_historial(self, monkeypatch):
        llamadas = []

        def falso(historial, pregunta, idioma_destino=None):
            llamadas.append((historial, pregunta, idioma_destino))
            return f"EN: {pregunta}", True

        monkeypatch.setattr(lc, "reformular_consulta", falso)
        casos = [{"pregunta": "hola", "fuente": "m.pdf", "paginas": {1}}]

        resultado = traducir_casos(casos, "en")

        assert llamadas == [("", "hola", "en")]
        assert resultado[0]["pregunta"] == "EN: hola"
        assert resultado[0]["original"] == "hola"
        assert resultado[0]["traduccion_ok"] is True
        assert casos[0]["pregunta"] == "hola"  # no muta la entrada

    def test_sin_api_key_devuelve_error(self, archivos, monkeypatch, capsys):
        carpeta, preguntas = archivos
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

        codigo = main(["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir"])

        assert codigo == 1
        assert "GROQ_API_KEY" in capsys.readouterr().err

    def test_sin_idioma_claro_devuelve_error(self, archivos, monkeypatch, capsys):
        carpeta, preguntas = archivos
        monkeypatch.setenv("GROQ_API_KEY", "clave")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

        codigo = main(["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir"])

        assert codigo == 1
        assert "idioma" in capsys.readouterr().err

    def test_con_idioma_traduce_y_lo_muestra_en_verbose(self, archivos, monkeypatch, capsys):
        carpeta, preguntas = archivos
        monkeypatch.setenv("GROQ_API_KEY", "clave")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        monkeypatch.setattr(IndiceDocumental, "idioma_principal", property(lambda self: "en"))
        monkeypatch.setattr(
            lc, "reformular_consulta", lambda h, p, idioma_destino=None: ("What is E-42?", True)
        )

        codigo = main(
            ["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir", "--verbose"]
        )

        salida = capsys.readouterr().out
        assert codigo == 0
        assert "Traducciones usadas para buscar" in salida
        assert "What is E-42?" in salida and "¿Qué es E-42?" in salida

    def preparar(self, monkeypatch, respuestas):
        """`respuestas`: función pregunta -> (texto, ok), en lugar del LLM."""
        monkeypatch.setenv("GROQ_API_KEY", "clave")
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        monkeypatch.setattr(IndiceDocumental, "idioma_principal", property(lambda self: "en"))
        monkeypatch.setattr(
            lc, "reformular_consulta", lambda h, p, idioma_destino=None: respuestas(p)
        )

    def test_si_la_traduccion_falla_en_todas_se_aborta_sin_informe(
        self, archivos, monkeypatch, capsys
    ):
        # Lo que pasó de verdad: un modelo inexistente falla en cada pregunta. Evaluar con
        # las preguntas originales daría un informe que parece de la traducción y no lo es.
        carpeta, preguntas = archivos
        self.preparar(monkeypatch, lambda p: (p, False))

        codigo = main(["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir"])

        captura = capsys.readouterr()
        assert codigo == 1
        assert "GROQ_REFORMULATOR_MODEL" in captura.err
        assert "Hit@1" not in captura.out

    def test_fallo_parcial_avisa_y_sigue(self, tmp_path, archivos, monkeypatch, capsys):
        carpeta, _ = archivos
        preguntas = tmp_path / "dos.json"
        preguntas.write_text(
            json.dumps(
                [
                    {"pregunta": "uno", "fuente": "manual.pdf", "paginas": [1]},
                    {"pregunta": "dos", "fuente": "manual.pdf", "paginas": [1]},
                ]
            ),
            encoding="utf-8",
        )
        self.preparar(monkeypatch, lambda p: (p, p != "dos"))

        codigo = main(["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir"])

        captura = capsys.readouterr()
        assert codigo == 0
        assert "1 de 2" in captura.err
        assert "Hit@1" in captura.out

    def test_pregunta_ya_en_el_idioma_de_destino_no_es_un_fallo(
        self, archivos, monkeypatch, capsys
    ):
        carpeta, preguntas = archivos
        self.preparar(monkeypatch, lambda p: (p, True))  # devuelta igual, pero sin error

        codigo = main(["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--traducir"])

        assert codigo == 0
        assert "no se pudieron traducir" not in capsys.readouterr().err