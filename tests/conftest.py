"""Fixtures compartidas. Ningún test descarga modelos ni llama a Groq."""

import hashlib
import io
import math

import pytest
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from backend import document_processor as dp


class EmbeddingsFalsos(Embeddings):
    """Bolsa de palabras hasheada: determinista, instantánea y con cierta semántica léxica."""

    DIM = 64

    def _vector(self, texto: str) -> list[float]:
        v = [0.0] * self.DIM
        for token in texto.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            v[idx] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class ArchivoFalso:
    """Imita el UploadedFile de Streamlit (solo .name y .getvalue())."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _pdf_con_texto(paginas: list[str]) -> bytes:
    """Genera un PDF mínimo válido con una línea de texto ASCII por página."""
    n = len(paginas)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objetos = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, texto in enumerate(paginas):
        objetos.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        flujo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET"
        objetos.append(f"<< /Length {len(flujo)} >>\nstream\n{flujo}\nendstream")

    salida = b"%PDF-1.4\n"
    offsets = []
    for numero, cuerpo in enumerate(objetos, start=1):
        offsets.append(len(salida))
        salida += f"{numero} 0 obj\n{cuerpo}\nendobj\n".encode()

    xref = len(salida)
    salida += f"xref\n0 {len(objetos) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        salida += f"{offset:010d} 00000 n \n".encode()
    salida += (
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return salida


@pytest.fixture(autouse=True)
def embeddings_falsos(monkeypatch):
    monkeypatch.setattr(dp, "obtener_embeddings", lambda: EmbeddingsFalsos())


@pytest.fixture(autouse=True)
def indice_temporal(monkeypatch, tmp_path):
    """Ningún test debe leer ni escribir el índice real del proyecto."""
    monkeypatch.setattr(dp, "INDEX_DIR", tmp_path / "indice_por_defecto")


@pytest.fixture
def crear_pdf():
    return _pdf_con_texto


@pytest.fixture
def pdf_sin_texto() -> bytes:
    escritor = PdfWriter()
    escritor.add_blank_page(612, 792)
    buffer = io.BytesIO()
    escritor.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def archivo():
    return ArchivoFalso
