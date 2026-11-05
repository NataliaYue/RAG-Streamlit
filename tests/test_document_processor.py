from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend import document_processor as dp
from backend.document_processor import (
    IndiceDocumental,
    construir_contexto,
    formatear_fuentes,
)


def doc(fuente: str, pagina: int, texto: str = "texto") -> Document:
    return Document(page_content=texto, metadata={"source": fuente, "page": pagina})


PAGINAS_MANUAL = [
    "El error E-42 indica sobrecalentamiento del motor",
    "Instrucciones de limpieza del filtro cada semana",
    "Garantia y condiciones de devolucion del producto",
]


# --- construir_contexto / formatear_fuentes -----------------------------------
class TestUtilidades:
    def test_contexto_incluye_procedencia_y_texto(self):
        contexto = construir_contexto([doc("a.pdf", 3, "hola mundo")])
        assert 'fuente="a.pdf"' in contexto
        assert 'pagina="3"' in contexto
        assert "hola mundo" in contexto

    def test_contexto_vacio(self):
        assert construir_contexto([]) == ""

    def test_contexto_separa_fragmentos(self):
        contexto = construir_contexto([doc("a.pdf", 1), doc("b.pdf", 2)])
        assert contexto.count("<fragmento") == 2
        assert contexto.count("</fragmento>") == 2

    def test_fuentes_agrupa_ordena_y_deduplica(self):
        docs = [doc("a.pdf", 3), doc("a.pdf", 1), doc("b.pdf", 2), doc("a.pdf", 3)]
        assert formatear_fuentes(docs) == "**Fuentes:** a.pdf (p. 1, 3); b.pdf (p. 2)"

    def test_fuentes_vacias(self):
        assert formatear_fuentes([]) == ""


# --- Extracción de PDF -----------------------------------------------------------
class TestExtraccion:
    def test_una_document_por_pagina_con_metadatos(self, crear_pdf):
        paginas = dp._extraer_paginas(
            crear_pdf(["Primera pagina", "Segunda pagina"]), "m.pdf"
        )
        assert len(paginas) == 2
        assert [p.metadata["page"] for p in paginas] == [1, 2]
        assert all(p.metadata["source"] == "m.pdf" for p in paginas)
        assert "Primera" in paginas[0].page_content

    def test_pdf_sin_texto_devuelve_lista_vacia(self, pdf_sin_texto):
        assert dp._extraer_paginas(pdf_sin_texto, "escaneado.pdf") == []

    def test_bytes_invalidos_lanzan_valueerror(self):
        with pytest.raises(ValueError, match="No se pudo leer"):
            dp._extraer_paginas(b"esto no es un pdf", "roto.pdf")


# --- Indexado ----------------------------------------------------------------------
class TestIndexado:
    def test_indice_nuevo_esta_vacio(self):
        indice = IndiceDocumental()
        assert indice.vacio
        assert indice.n_documentos == 0
        assert indice.n_fragmentos == 0
        assert indice.nombres == []

    def test_indexa_un_archivo(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos([archivo("a.pdf", crear_pdf(PAGINAS_MANUAL))])
        assert avisos == []
        assert not indice.vacio
        assert indice.n_documentos == 1
        assert indice.n_fragmentos >= 3
        assert indice.nombres == ["a.pdf"]

    def test_indexado_incremental_conserva_lo_anterior(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos(
            [archivo("a.pdf", crear_pdf(["Contenido del manual A"]))]
        )
        fragmentos_a = indice.n_fragmentos

        indice.agregar_archivos(
            [archivo("b.pdf", crear_pdf(["Contenido del manual B"]))]
        )

        assert indice.n_documentos == 2
        assert indice.n_fragmentos == fragmentos_a + 1
        assert indice.nombres == ["a.pdf", "b.pdf"]

    def test_duplicado_por_contenido_se_omite_aunque_cambie_el_nombre(
        self, archivo, crear_pdf
    ):
        datos = crear_pdf(["Mismo contenido"])
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("a.pdf", datos)])

        avisos = indice.agregar_archivos([archivo("copia.pdf", datos)])

        assert any("ya estaba indexado" in a for a in avisos)
        assert indice.n_documentos == 1

    def test_duplicado_dentro_del_mismo_lote(self, archivo, crear_pdf):
        datos = crear_pdf(["Mismo contenido"])
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos(
            [archivo("a.pdf", datos), archivo("b.pdf", datos)]
        )
        assert indice.n_documentos == 1
        assert any("ya estaba indexado" in a for a in avisos)

    def test_pdf_sin_texto_avisa_y_no_indexa(self, archivo, pdf_sin_texto):
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos([archivo("escaneado.pdf", pdf_sin_texto)])
        assert any("no contiene texto" in a for a in avisos)
        assert indice.vacio

    def test_archivo_corrupto_avisa_y_no_rompe(self, archivo):
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos([archivo("roto.pdf", b"basura")])
        assert any("No se pudo leer" in a for a in avisos)
        assert indice.vacio

    def test_un_archivo_malo_no_bloquea_a_los_buenos(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos(
            [
                archivo("roto.pdf", b"basura"),
                archivo("ok.pdf", crear_pdf(["Texto valido"])),
            ]
        )
        assert len(avisos) == 1
        assert indice.nombres == ["ok.pdf"]

    def test_limite_de_tamano(self, archivo, crear_pdf, monkeypatch):
        monkeypatch.setattr(dp, "MAX_MB_POR_ARCHIVO", 0)
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos([archivo("grande.pdf", crear_pdf(["Texto"]))])
        assert any("supera el límite" in a for a in avisos)
        assert indice.vacio

    def test_troceado_conserva_metadatos(self, archivo, crear_pdf):
        texto_largo = "palabra " * 600  # ~4800 caracteres, más que CHUNK_SIZE
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("largo.pdf", crear_pdf([texto_largo]))])

        assert indice.n_fragmentos > 1
        for frag in indice.fragmentos:
            assert frag.metadata["source"] == "largo.pdf"
            assert frag.metadata["page"] == 1


# --- Búsqueda híbrida -----------------------------------------------------------------
class TestBusqueda:
    @pytest.fixture
    def indice(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("manual.pdf", crear_pdf(PAGINAS_MANUAL))])
        return indice

    def test_indice_vacio_devuelve_lista_vacia(self):
        assert IndiceDocumental().buscar("lo que sea") == []

    def test_encuentra_codigo_exacto(self, indice):
        resultado = indice.buscar("E-42", k=1)
        assert len(resultado) == 1
        assert resultado[0].metadata["page"] == 1

    def test_respeta_k(self, indice):
        assert len(indice.buscar("motor filtro garantia", k=2)) == 2

    def test_k_mayor_que_fragmentos_no_falla(self, indice):
        assert len(indice.buscar("motor", k=50)) <= indice.n_fragmentos

    def test_sin_duplicados(self, indice):
        resultado = indice.buscar("motor filtro garantia", k=10)
        claves = [(d.metadata["page"], d.page_content) for d in resultado]
        assert len(claves) == len(set(claves))

    def test_fusion_rrf_premia_lo_que_aparece_en_ambas_listas(self):
        a, b, c = doc("x.pdf", 1, "a"), doc("x.pdf", 2, "b"), doc("x.pdf", 3, "c")
        indice = IndiceDocumental()
        indice.fragmentos = [a, b, c]
        indice.vector_store = SimpleNamespace(similarity_search=lambda q, k: [a, b])
        indice._bm25 = SimpleNamespace(invoke=lambda q: [c, b])

        resultado = indice.buscar("consulta", k=3)

        assert resultado[0] is b  # aparece en las dos listas
        assert {id(d) for d in resultado} == {id(a), id(b), id(c)}  # combina ambas

    def test_modo_lexico_encuentra_codigo_exacto(self, indice):
        resultado = indice.buscar("E-42", k=1, modo="lexico")
        assert resultado[0].metadata["page"] == 1

    def test_modo_semantico_respeta_k(self, indice):
        assert len(indice.buscar("motor", k=2, modo="semantico")) == 2

    def test_modo_desconocido_lanza_error(self, indice):
        with pytest.raises(ValueError, match="Modo de búsqueda"):
            indice.buscar("motor", modo="inventado")
