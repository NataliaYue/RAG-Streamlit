import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from backend.document_processor import IndiceDocumental, parsear_rangos
from evaluation.evaluar import main

PAGINAS = [
    "El error E-42 indica sobrecalentamiento del motor",
    "Instrucciones de limpieza del filtro cada semana",
    "Garantia y condiciones de devolucion del producto",
]


def doc(pagina: int, texto: str) -> Document:
    return Document(page_content=texto, metadata={"source": "x.pdf", "page": pagina})


# --- parsear_rangos ---------------------------------------------------------------------
class TestParsearRangos:
    @pytest.mark.parametrize("vacio", [None, "", "   ", " , ; "])
    def test_vacio(self, vacio):
        assert parsear_rangos(vacio) == set()

    def test_numero_suelto_y_rango(self):
        assert parsear_rangos("5, 8-10") == {5, 8, 9, 10}

    def test_separador_punto_y_coma_y_espacios(self):
        assert parsear_rangos("1 - 3; 7") == {1, 2, 3, 7}

    def test_rango_abierto_llega_muy_lejos(self):
        paginas = parsear_rangos("125-")
        assert 125 in paginas and 148 in paginas and 124 not in paginas

    @pytest.mark.parametrize("malo", ["abc", "3-1", "0", "-5", "1-2-3", "2,x"])
    def test_invalido_lanza_error(self, malo):
        with pytest.raises(ValueError, match="no válido"):
            parsear_rangos(malo)


# --- Exclusión al indexar --------------------------------------------------------------------
class TestExclusion:
    def test_paginas_excluidas_no_se_indexan(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos(
            [archivo("m.pdf", crear_pdf(PAGINAS))], excluir_paginas={3}
        )
        assert {f.metadata["page"] for f in indice.fragmentos} == {1, 2}

    def test_excluida_no_aparece_en_ninguna_busqueda(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos(
            [archivo("m.pdf", crear_pdf(PAGINAS))], excluir_paginas={3}
        )
        for modo in ("hibrido", "semantico", "lexico"):
            paginas = {d.metadata["page"] for d in indice.buscar("garantia devolucion", k=5, modo=modo)}
            assert 3 not in paginas

    def test_sin_excluir_se_indexa_todo(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos([archivo("m.pdf", crear_pdf(PAGINAS))])
        assert {f.metadata["page"] for f in indice.fragmentos} == {1, 2, 3}

    def test_excluir_todo_avisa_y_no_indexa(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        avisos = indice.agregar_archivos(
            [archivo("m.pdf", crear_pdf(PAGINAS))], excluir_paginas={1, 2, 3}
        )
        assert indice.vacio
        assert any("tras excluir" in a for a in avisos)

    def test_excluir_conserva_numeracion_original(self, archivo, crear_pdf):
        indice = IndiceDocumental()
        indice.agregar_archivos(
            [archivo("m.pdf", crear_pdf(PAGINAS))], excluir_paginas={1}
        )
        assert {f.metadata["page"] for f in indice.fragmentos} == {2, 3}


# --- Peso de BM25 en la fusión --------------------------------------------------------------------
class TestPesoLexico:
    @pytest.fixture
    def indice(self):
        a, b, c = doc(1, "a"), doc(2, "b"), doc(3, "c")
        indice = IndiceDocumental()
        indice.fragmentos = [a, b, c]
        indice.vector_store = SimpleNamespace(similarity_search=lambda q, k: [a, b])
        indice._bm25 = SimpleNamespace(invoke=lambda q: [c])
        return indice

    def test_peso_alto_deja_entrar_a_bm25(self, indice):
        paginas = [d.metadata["page"] for d in indice.buscar("q", k=2, peso_lexico=1.0)]
        assert 3 in paginas

    def test_peso_bajo_prioriza_lo_semantico(self, indice):
        paginas = [d.metadata["page"] for d in indice.buscar("q", k=2, peso_lexico=0.1)]
        assert paginas == [1, 2]

    def test_peso_por_defecto_viene_de_la_configuracion(self, indice, monkeypatch):
        from backend import document_processor as dp

        monkeypatch.setattr(dp, "PESO_LEXICO", 0.1)
        assert [d.metadata["page"] for d in indice.buscar("q", k=2)] == [1, 2]

    def test_peso_cero_equivale_al_orden_semantico(self, indice):
        paginas = [d.metadata["page"] for d in indice.buscar("q", k=3, peso_lexico=0.0)]
        assert paginas[:2] == [1, 2]


# --- CLI de evaluación -------------------------------------------------------------------------------
class TestCLI:
    @pytest.fixture
    def entorno(self, tmp_path, crear_pdf):
        carpeta = tmp_path / "pdfs"
        carpeta.mkdir()
        (carpeta / "manual.pdf").write_bytes(crear_pdf(PAGINAS))
        preguntas = tmp_path / "preguntas.json"
        preguntas.write_text(
            json.dumps([{"pregunta": "E-42", "fuente": "manual.pdf", "paginas": [1]}]),
            encoding="utf-8",
        )
        return carpeta, preguntas

    def test_excluir_paginas_reduce_los_fragmentos(self, entorno, capsys):
        carpeta, preguntas = entorno
        codigo = main(
            ["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--excluir-paginas", "3"]
        )
        assert codigo == 0
        # 3 páginas cortas -> 3 fragmentos; sin la 3 quedan 2
        assert "2 fragmentos" in capsys.readouterr().out

    def test_pesos_lexico_anade_filas_al_informe(self, entorno, capsys):
        carpeta, preguntas = entorno
        codigo = main(
            ["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--k", "3",
             "--pesos-lexico", "0", "0.3", "1"]
        )
        salida = capsys.readouterr().out
        assert codigo == 0
        for etiqueta in ("hib w=0", "hib w=0.3", "hib w=1"):
            assert etiqueta in salida

    def test_rango_invalido_devuelve_error(self, entorno, capsys):
        carpeta, preguntas = entorno
        codigo = main(
            ["--pdfs", str(carpeta), "--preguntas", str(preguntas), "--excluir-paginas", "abc"]
        )
        assert codigo == 1
        assert "no válido" in capsys.readouterr().err