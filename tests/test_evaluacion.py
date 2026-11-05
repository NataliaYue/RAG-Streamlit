import pytest

from backend import document_processor as dp
from backend.document_processor import IndiceDocumental

PAGINAS = [
    "El error E-42 indica sobrecalentamiento del motor",
    "Instrucciones de limpieza del filtro cada semana",
    "Garantia y condiciones de devolucion del producto",
]


@pytest.fixture
def indice(archivo, crear_pdf):
    indice = IndiceDocumental()
    indice.agregar_archivos([archivo("manual.pdf", crear_pdf(PAGINAS))])
    return indice


def test_guardar_y_cargar_conserva_todo(indice, tmp_path):
    indice.guardar(tmp_path)

    cargado = IndiceDocumental.cargar(tmp_path)

    assert cargado.nombres == ["manual.pdf"]
    assert cargado.n_documentos == 1
    assert cargado.n_fragmentos == indice.n_fragmentos
    resultado = cargado.buscar("E-42", k=1)
    assert [(d.metadata["source"], d.metadata["page"]) for d in resultado] == [
        ("manual.pdf", 1)
    ]


def test_cargar_directorio_inexistente_devuelve_indice_vacio(tmp_path):
    assert IndiceDocumental.cargar(tmp_path / "no_existe").vacio


def test_duplicados_se_detectan_tras_cargar(archivo, crear_pdf, tmp_path):
    datos = crear_pdf(PAGINAS)
    original = IndiceDocumental()
    original.agregar_archivos([archivo("manual.pdf", datos)])
    original.guardar(tmp_path)

    cargado = IndiceDocumental.cargar(tmp_path)
    avisos = cargado.agregar_archivos([archivo("copia.pdf", datos)])

    assert any("ya estaba indexado" in a for a in avisos)
    assert cargado.n_documentos == 1


def test_se_puede_seguir_indexando_tras_cargar(indice, archivo, crear_pdf, tmp_path):
    indice.guardar(tmp_path)
    cargado = IndiceDocumental.cargar(tmp_path)
    fragmentos_antes = cargado.n_fragmentos

    cargado.agregar_archivos(
        [archivo("otro.pdf", crear_pdf(["Contenido distinto del segundo manual"]))]
    )

    assert cargado.n_documentos == 2
    assert cargado.n_fragmentos == fragmentos_antes + 1
    assert cargado.nombres == ["manual.pdf", "otro.pdf"]


def test_modelo_de_embeddings_distinto_falla_con_mensaje_claro(
    indice, tmp_path, monkeypatch
):
    indice.guardar(tmp_path)
    monkeypatch.setattr(dp, "EMBEDDING_MODEL", "otro-modelo")

    with pytest.raises(ValueError, match="otro modelo"):
        IndiceDocumental.cargar(tmp_path)


def test_metadatos_corruptos_lanzan_valueerror(tmp_path):
    (tmp_path / "indice.json").write_text("esto no es json", encoding="utf-8")
    with pytest.raises(ValueError):
        IndiceDocumental.cargar(tmp_path)


def test_guardar_indice_vacio_borra_lo_guardado(indice, tmp_path):
    indice.guardar(tmp_path)
    assert (tmp_path / "indice.json").exists()

    IndiceDocumental().guardar(tmp_path)

    assert not (tmp_path / "indice.json").exists()
    assert IndiceDocumental.cargar(tmp_path).vacio


def test_eliminar_no_toca_archivos_ajenos(indice, tmp_path):
    ajeno = tmp_path / "notas.txt"
    ajeno.write_text("importante")
    indice.guardar(tmp_path)

    IndiceDocumental.eliminar(tmp_path)

    assert ajeno.exists()
    assert not (tmp_path / "indice.json").exists()
    assert not (tmp_path / "index.faiss").exists()


def test_eliminar_directorio_inexistente_no_falla(tmp_path):
    IndiceDocumental.eliminar(tmp_path / "no_existe")


def test_sin_argumento_usa_index_dir(indice, tmp_path, monkeypatch):
    destino = tmp_path / "por_defecto"
    monkeypatch.setattr(dp, "INDEX_DIR", destino)

    indice.guardar()

    assert (destino / "indice.json").exists()
    assert IndiceDocumental.cargar().n_fragmentos == indice.n_fragmentos
