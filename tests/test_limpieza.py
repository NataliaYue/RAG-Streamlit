from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend import document_processor as dp
from backend import limpieza as L
from backend.document_processor import IndiceDocumental

EN = (
    "Before you start the procedure make sure that your computer is turned off and that "
    "you have read all of the safety information in this guide because it is important"
)
ES = (
    "Antes de empezar el procedimiento asegurese de que el equipo esta apagado y de que "
    "ha leido toda la informacion de seguridad de esta guia porque es muy importante para usted"
)
PT = (
    "Antes de iniciar o procedimento certifique se de que o computador esta desligado e que "
    "voce leu todas as informacoes de seguranca deste guia porque elas são muito importantes para os usuarios"
)
DE = (
    "Bevor sie mit dem verfahren beginnen stellen sie sicher dass der computer ausgeschaltet ist "
    "und dass sie alle sicherheitshinweise in diesem handbuch gelesen haben weil sie nicht unwichtig sind"
)
INDICE_TXT = "\n".join(f"Capitulo {i} del manual . . . . . . . . {i * 3}" for i in range(1, 11))


def pagina(n: int, texto: str, fuente: str = "m.pdf") -> Document:
    return Document(page_content=texto, metadata={"source": fuente, "page": n})


def ruidos(paginas):
    return {p.metadata["page"]: p.metadata.get("ruido") for p in paginas}


# --- Estructura: índices y apéndices -------------------------------------------------------
class TestMarcarPaginas:
    def test_apendice_dura_hasta_el_siguiente_capitulo(self):
        paginas = [
            pagina(1, "Chapter 1. Product overview\ncontenido"),
            pagina(2, "texto normal del manual"),
            pagina(3, "Appendix A. Supplemental information\ntexto"),
            pagina(4, "mas texto sin encabezado"),
            pagina(5, "Chapter 2. Using your computer\ncontenido"),
            pagina(6, "texto normal otra vez"),
        ]
        L.marcar_paginas(paginas)
        assert ruidos(paginas) == {
            1: None, 2: None, 3: L.APENDICE, 4: L.APENDICE, 5: None, 6: None,
        }

    def test_apendice_sin_capitulo_posterior_llega_al_final(self):
        paginas = [pagina(1, "Chapter 1. Intro"), pagina(2, "Appendix B. Regulatory information"),
                   pagina(3, "texto"), pagina(4, "texto")]
        L.marcar_paginas(paginas)
        assert [ruidos(paginas)[n] for n in (1, 2, 3, 4)] == [
            None, L.APENDICE, L.APENDICE, L.APENDICE,
        ]

    def test_indice_se_detecta_por_puntos_guia(self):
        paginas = [pagina(1, INDICE_TXT), pagina(2, "texto normal")]
        L.marcar_paginas(paginas)
        assert ruidos(paginas) == {1: L.INDICE, 2: None}

    def test_un_indice_que_lista_apendices_no_abre_seccion_final(self):
        # Caso real: el índice empieza por "Appendix C..." y las páginas siguientes son seguridad.
        indice = "Appendix C. WEEE and recycling . . . . 115\n" + INDICE_TXT
        paginas = [pagina(1, indice), pagina(2, "Important safety information\ntexto util"),
                   pagina(3, "mas seguridad"), pagina(4, "Chapter 1. Overview")]
        L.marcar_paginas(paginas)
        assert ruidos(paginas) == {1: L.INDICE, 2: None, 3: None, 4: None}

    def test_notice_en_singular_no_es_seccion_final(self):
        paginas = [pagina(1, "Notice for built-in rechargeable battery\ntexto")]
        L.marcar_paginas(paginas)
        assert ruidos(paginas) == {1: None}

    @pytest.mark.parametrize(
        "encabezado",
        ["Anexo A. Informacion adicional", "Apéndice B. Normativa", "Annex C. Compliance",
         "Notices\ntexto", "Trademarks\ntexto", "Avisos legales\ntexto"],
    )
    def test_encabezados_de_seccion_final_en_varios_idiomas(self, encabezado):
        paginas = [pagina(1, "Chapter 1. Intro"), pagina(2, encabezado)]
        L.marcar_paginas(paginas)
        assert ruidos(paginas)[2] == L.APENDICE

    def test_manual_sin_marcadores_no_marca_nada(self):
        paginas = [pagina(n, f"texto normal de la pagina {n}") for n in range(1, 6)]
        L.marcar_paginas(paginas)
        assert set(ruidos(paginas).values()) == {None}


# --- Idioma --------------------------------------------------------------------------------------
class TestDetectarIdioma:
    @pytest.mark.parametrize(
        "texto, esperado", [(EN, "en"), (ES, "es"), (PT, "pt"), (DE, "de")]
    )
    def test_idiomas_largos(self, texto, esperado):
        assert L.detectar_idioma(texto) == esperado

    def test_texto_corto_no_concluyente(self):
        assert L.detectar_idioma("the error code") is None

    def test_texto_sin_palabras_funcionales(self):
        assert L.detectar_idioma("E-42 0183 0187 0188 2201 " * 10) is None


class TestMarcarIdiomas:
    def frags(self, textos, fuente="m.pdf"):
        return [pagina(i + 1, t, fuente) for i, t in enumerate(textos)]

    def test_minoritario_se_marca_y_el_principal_no(self):
        frs = self.frags([EN] * 25 + [DE] * 2)
        L.marcar_idiomas(frs)
        marcados = [f.metadata["page"] for f in frs if f.metadata.get("ruido") == L.IDIOMA]
        assert marcados == [26, 27]

    def test_manual_bilingue_no_se_toca(self):
        frs = self.frags([EN] * 15 + [ES] * 15)
        L.marcar_idiomas(frs)
        assert not any(f.metadata.get("ruido") for f in frs)

    def test_pocos_fragmentos_no_se_marca_nada(self):
        frs = self.frags([EN] * 5 + [DE])
        L.marcar_idiomas(frs)
        assert not any(f.metadata.get("ruido") for f in frs)

    def test_no_pisa_un_marcado_previo(self):
        frs = self.frags([EN] * 25 + [DE])
        frs[-1].metadata["ruido"] = L.APENDICE
        L.marcar_idiomas(frs)
        assert frs[-1].metadata["ruido"] == L.APENDICE

    def test_cada_documento_tiene_su_propio_idioma_principal(self):
        frs = self.frags([EN] * 25, "a.pdf") + self.frags([ES] * 25, "b.pdf")
        L.marcar_idiomas(frs)
        assert not any(f.metadata.get("ruido") for f in frs)


# --- Penalización en la búsqueda ----------------------------------------------------------------
class TestPenalizacion:
    @pytest.fixture
    def indice(self):
        a = Document(page_content="a", metadata={"source": "x.pdf", "page": 1, "ruido": L.APENDICE})
        b = Document(page_content="b", metadata={"source": "x.pdf", "page": 2})
        indice = IndiceDocumental()
        indice.fragmentos = [a, b]
        indice.vector_store = SimpleNamespace(similarity_search=lambda q, k: [a, b])
        indice._bm25 = SimpleNamespace(invoke=lambda q: [a, b])
        return indice

    def paginas(self, indice, **kw):
        return [d.metadata["page"] for d in indice.buscar("q", k=2, **kw)]

    @pytest.mark.parametrize("modo", ["hibrido", "semantico", "lexico"])
    def test_sin_penalizar_manda_el_orden_original(self, indice, modo):
        assert self.paginas(indice, modo=modo, penalizacion_ruido=1.0) == [1, 2]

    @pytest.mark.parametrize("modo", ["hibrido", "semantico", "lexico"])
    def test_penalizar_baja_el_marcado_en_todos_los_modos(self, indice, modo):
        assert self.paginas(indice, modo=modo, penalizacion_ruido=0.7) == [2, 1]

    def test_lo_marcado_sigue_ganando_si_es_claramente_mejor(self, indice):
        # Primero en la lista semántica y único en BM25 (a peso completo): la suma de ambas
        # señales supera a la penalización. Un resultado solo mediocre no lo lograría.
        a = indice.fragmentos[0]
        indice._bm25 = SimpleNamespace(invoke=lambda q: [a])
        assert self.paginas(indice, peso_lexico=1.0, penalizacion_ruido=0.7) == [1, 2]

    def test_el_valor_por_defecto_viene_de_la_configuracion(self, indice, monkeypatch):
        monkeypatch.setattr(dp, "PENALIZACION_RUIDO", 1.0)
        assert self.paginas(indice, modo="semantico") == [1, 2]
        monkeypatch.setattr(dp, "PENALIZACION_RUIDO", 0.5)
        assert self.paginas(indice, modo="semantico") == [2, 1]

    def test_n_ruido(self, indice):
        assert indice.n_ruido == 1


# --- Integración: PDF -> marcado -> disco -> búsqueda ---------------------------------------------
# Textos elegidos para que los embeddings simulados (bolsa de palabras hasheada a 64
# dimensiones) no colisionen y la página del apéndice sea la más parecida a la pregunta.
PAGINAS = [
    "garantia producto manual",
    "Appendix A garantia devolucion producto",
]


class TestIntegracion:
    @pytest.fixture
    def indice(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("m.pdf", crear_pdf(PAGINAS))])
        return indice

    def test_el_apendice_se_marca_al_indexar(self, indice):
        marcados = {f.metadata["page"] for f in indice.fragmentos if f.metadata.get("ruido")}
        assert marcados == {2}
        assert indice.n_ruido == 1

    def test_el_marcado_sobrevive_a_guardar_y_cargar(self, indice, tmp_path):
        indice.guardar(tmp_path)
        cargado = IndiceDocumental.cargar(tmp_path)
        assert cargado.n_ruido == 1
        marcados = {d.metadata["page"] for d in cargado.vector_store.docstore._dict.values()
                    if d.metadata.get("ruido")}
        assert marcados == {2}

    def test_penalizar_cambia_el_orden_del_resultado(self, indice):
        pregunta = "garantia devolucion producto"  # coincide casi literal con el apendice
        sin = indice.buscar(pregunta, k=2, modo="semantico", penalizacion_ruido=1.0)
        con = indice.buscar(pregunta, k=2, modo="semantico", penalizacion_ruido=0.7)
        assert sin[0].metadata["page"] == 2
        assert con[0].metadata["page"] == 1

    def test_excluir_y_marcar_conviven(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        paginas = ["Chapter 1 uso normal", "texto normal", "Appendix A avisos", "texto legal"]
        indice.agregar_archivos([archivo("m.pdf", crear_pdf(paginas))], excluir_paginas={2})
        assert {f.metadata["page"] for f in indice.fragmentos} == {1, 3, 4}
        assert indice.n_ruido == 2