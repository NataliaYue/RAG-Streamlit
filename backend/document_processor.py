"""Ingesta de PDFs, indexado incremental y recuperación híbrida (FAISS + BM25)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.limpieza import (
    MIN_FRAGMENTOS_IDIOMA,
    marcar_idiomas,
    marcar_paginas,
    tokenizar_bm25,
)

logger = logging.getLogger(__name__)

# --- Configuración (sobrescribible con variables de entorno) -----------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))  # fragmentos que llegan al LLM
FETCH_K = int(os.getenv("FETCH_K", "20"))  # candidatos de cada buscador antes de fusionar
MAX_MB_POR_ARCHIVO = int(os.getenv("MAX_MB_POR_ARCHIVO", "50"))
RRF_K = 60  # constante estándar de Reciprocal Rank Fusion
# Peso de BM25 frente a la búsqueda semántica (1.0) en la fusión. Con el tokenizador actual
# y consulta e idioma del manual coincidentes, 1.0 fue lo mejor en la evaluación (Hit@5 1.00
# frente a 0.89 con 0.3). Bájalo (p. ej. PESO_LEXICO=0.3) si buscas en un idioma distinto
# al de los manuales sin traducir la consulta: ahí BM25 solo acierta por casualidad.
PESO_LEXICO = float(os.getenv("PESO_LEXICO", "1.0"))
# Peso de BM25 cuando se esperaba traducir la consulta y la traducción falló: se busca con
# la pregunta en otro idioma que el manual y BM25 solo acierta por casualidad (en la
# evaluación, el híbrido con BM25 completo quedó por debajo del semántico: .44 frente a .56).
PESO_LEXICO_SIN_TRADUCCION = float(os.getenv("PESO_LEXICO_SIN_TRADUCCION", "0.0"))
# Factor que multiplica la puntuación de los fragmentos marcados como de bajo valor
# (índices, apéndices, otro idioma; ver limpieza.py). 1.0 = no penalizar. Es un factor
# sobre una puntuación por posición: 0.7 solo deja pasar a un fragmento marcado si es
# claramente de lo mejor que se ha encontrado.
PENALIZACION_RUIDO = float(os.getenv("PENALIZACION_RUIDO", "0.7"))
# Para traducir la consulta, un idioma debe reunir al menos esta parte de los fragmentos.
DOMINANCIA_IDIOMA = float(os.getenv("DOMINANCIA_IDIOMA", "0.7"))

# Carpeta del índice guardado (por defecto <raíz del proyecto>/data/index)
INDEX_DIR = Path(
    os.getenv("INDEX_DIR", Path(__file__).resolve().parent.parent / "data" / "index")
)
MODOS_BUSQUEDA = ("hibrido", "semantico", "lexico")
_ARCHIVOS_INDICE = ("index.faiss", "index.pkl", "indice.json")
_VERSION_INDICE = 1


@lru_cache(maxsize=1)
def obtener_embeddings() -> HuggingFaceEmbeddings:
    """El modelo se carga una sola vez por proceso, no en cada indexado."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def parsear_rangos(texto: str | None) -> set[int]:
    """Convierte "125-148, 5, 200-" en un conjunto de páginas. Lanza ValueError si es inválido.

    Un rango abierto por la derecha ("125-") llega hasta la página 100000.
    """
    paginas: set[int] = set()
    for trozo in (texto or "").replace(";", ",").split(","):
        trozo = trozo.strip()
        if not trozo:
            continue
        try:
            if "-" in trozo:
                inicio, fin = (x.strip() for x in trozo.split("-", 1))
                a, b = int(inicio), int(fin) if fin else 100_000
            else:
                a = b = int(trozo)
        except ValueError:
            raise ValueError(f"Rango de páginas no válido: «{trozo}».") from None
        if a < 1 or b < a:
            raise ValueError(f"Rango de páginas no válido: «{trozo}».")
        paginas.update(range(a, b + 1))
    return paginas


def _extraer_paginas(
    contenido: bytes, nombre: str, excluir: set[int] | None = None
) -> list[Document]:
    """Devuelve un Document por página con texto. Lanza ValueError si el PDF no se puede leer."""
    excluir = excluir or set()
    try:
        lector = PdfReader(io.BytesIO(contenido))
        if lector.is_encrypted:
            lector.decrypt("")
        paginas = []
        for num_pagina, pagina in enumerate(lector.pages, start=1):
            texto = (pagina.extract_text() or "").strip()
            if texto:
                paginas.append(
                    Document(
                        page_content=texto,
                        metadata={"source": nombre, "page": num_pagina},
                    )
                )
        # El marcado va antes de excluir: la estructura (apéndices) se ve con todas las páginas.
        marcar_paginas(paginas)
        return [p for p in paginas if p.metadata["page"] not in excluir]
    except Exception as e:
        raise ValueError(f"No se pudo leer «{nombre}»: {e}") from e


def _crear_bm25(fragmentos: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(
        fragmentos, k=FETCH_K, preprocess_func=tokenizar_bm25
    )


def _clave(doc: Document) -> tuple:
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


class IndiceDocumental:
    """Índice acumulativo: vectores (FAISS) + léxico (BM25) sobre los mismos fragmentos."""

    def __init__(self) -> None:
        self.fragmentos: list[Document] = []
        self.vector_store: FAISS | None = None
        self._bm25: BM25Retriever | None = None
        self._documentos: dict[str, str] = {}  # hash del archivo -> nombre

    # --- Estado ---------------------------------------------------------------
    @property
    def vacio(self) -> bool:
        return not self.fragmentos

    @property
    def n_documentos(self) -> int:
        return len(self._documentos)

    @property
    def n_fragmentos(self) -> int:
        return len(self.fragmentos)

    @property
    def n_ruido(self) -> int:
        """Fragmentos marcados como de bajo valor (se rebajan en la búsqueda)."""
        return sum(1 for f in self.fragmentos if f.metadata.get("ruido"))

    @property
    def idioma_principal(self) -> str | None:
        """Idioma en el que está claramente escrito lo indexado, o None si no se puede decir.

        Exige suficientes fragmentos con idioma detectado y que uno domine con holgura;
        con manuales en varios idiomas devuelve None (y la consulta no se traduce).
        """
        conteo = Counter(
            f.metadata["idioma"] for f in self.fragmentos if f.metadata.get("idioma")
        )
        total = sum(conteo.values())
        if total < MIN_FRAGMENTOS_IDIOMA:
            return None
        idioma, n = conteo.most_common(1)[0]
        return idioma if n / total >= DOMINANCIA_IDIOMA else None

    @property
    def nombres(self) -> list[str]:
        return sorted(self._documentos.values())

    # --- Indexado ---------------------------------------------------------------
    def agregar_archivos(self, archivos, excluir_paginas: set[int] | None = None) -> list[str]:
        """Indexa los archivos nuevos y devuelve una lista de avisos para el usuario.

        `excluir_paginas` (números de página físicos, desde 1) deja fuera páginas como
        apéndices regulatorios o legales, que suelen estar en varios idiomas y se cuelan
        en los resultados. Se aplica a todos los archivos de la llamada.
        """
        avisos: list[str] = []
        paginas_nuevas: list[Document] = []
        hashes_nuevos: dict[str, str] = {}

        for archivo in archivos:
            contenido = archivo.getvalue()
            nombre = archivo.name

            if len(contenido) > MAX_MB_POR_ARCHIVO * 1024 * 1024:
                avisos.append(f"«{nombre}» supera el límite de {MAX_MB_POR_ARCHIVO} MB.")
                continue

            huella = hashlib.sha256(contenido).hexdigest()
            if huella in self._documentos or huella in hashes_nuevos:
                avisos.append(f"«{nombre}» ya estaba indexado; se ha omitido.")
                continue

            try:
                paginas = _extraer_paginas(contenido, nombre, excluir_paginas)
            except ValueError as e:
                logger.warning("%s", e)
                avisos.append(str(e))
                continue

            if not paginas:
                if excluir_paginas:
                    avisos.append(
                        f"«{nombre}» se ha omitido: no queda texto tras excluir las páginas indicadas."
                    )
                else:
                    avisos.append(
                        f"«{nombre}» no contiene texto extraíble (¿PDF escaneado?). "
                        "Necesita OCR antes de indexarse."
                    )
                continue

            paginas_nuevas.extend(paginas)
            hashes_nuevos[huella] = nombre

        if not paginas_nuevas:
            return avisos

        divisor = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        fragmentos = divisor.split_documents(paginas_nuevas)
        marcar_idiomas(fragmentos)  # antes de FAISS, para que el marcado se guarde con el índice

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(fragmentos, obtener_embeddings())
        else:
            self.vector_store.add_documents(fragmentos)

        self.fragmentos.extend(fragmentos)
        self._documentos.update(hashes_nuevos)
        self._bm25 = _crear_bm25(self.fragmentos)
        return avisos

    # --- Persistencia ---------------------------------------------------------------
    def guardar(self, directorio: Path | str | None = None) -> None:
        """Guarda el índice en disco. Un índice vacío borra lo que hubiera guardado."""
        directorio = Path(INDEX_DIR if directorio is None else directorio)
        if self.vacio or self.vector_store is None:
            self.eliminar(directorio)
            return

        directorio.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(directorio))  # index.faiss + index.pkl
        meta = {
            "version": _VERSION_INDICE,
            "modelo_embeddings": EMBEDDING_MODEL,
            "documentos": self._documentos,
            "fragmentos": [
                {"page_content": f.page_content, "metadata": f.metadata}
                for f in self.fragmentos
            ],
        }
        (directorio / "indice.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def cargar(cls, directorio: Path | str | None = None) -> "IndiceDocumental":
        """Carga un índice guardado. Si no hay ninguno, devuelve un índice vacío.

        Lanza ValueError si el índice está corrupto o se creó con otro modelo de embeddings.
        Usa pickle (FAISS): carga solo índices que hayas generado tú.
        """
        directorio = Path(INDEX_DIR if directorio is None else directorio)
        indice = cls()
        ruta_meta = directorio / "indice.json"
        if not ruta_meta.exists():
            return indice

        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        if meta.get("version") != _VERSION_INDICE:
            raise ValueError("El índice guardado tiene un formato incompatible; reindexa.")
        if meta.get("modelo_embeddings") != EMBEDDING_MODEL:
            raise ValueError(
                "El índice guardado se creó con otro modelo de embeddings "
                f"({meta.get('modelo_embeddings')}); reindexa los manuales."
            )

        indice.vector_store = FAISS.load_local(
            str(directorio), obtener_embeddings(), allow_dangerous_deserialization=True
        )
        indice.fragmentos = [
            Document(page_content=f["page_content"], metadata=f["metadata"])
            for f in meta["fragmentos"]
        ]
        indice._documentos = dict(meta["documentos"])
        if indice.fragmentos:
            indice._bm25 = _crear_bm25(indice.fragmentos)
        return indice

    @staticmethod
    def eliminar(directorio: Path | str | None = None) -> None:
        """Borra solo los archivos del índice (nunca el contenido ajeno del directorio)."""
        directorio = Path(INDEX_DIR if directorio is None else directorio)
        for nombre in _ARCHIVOS_INDICE:
            (directorio / nombre).unlink(missing_ok=True)
        try:
            directorio.rmdir()
        except OSError:
            pass  # no existe o contiene otros archivos

    # --- Recuperación -------------------------------------------------------------
    def buscar(
        self,
        pregunta: str,
        k: int = TOP_K,
        fetch_k: int = FETCH_K,
        modo: str = "hibrido",
        peso_lexico: float | None = None,
        penalizacion_ruido: float | None = None,
    ) -> list[Document]:
        """Búsqueda híbrida (semántica + BM25 fusionadas con RRF) o de un solo tipo.

        `modo` permite comparar estrategias en la evaluación: "hibrido" (por defecto),
        "semantico" o "lexico". `peso_lexico` es el peso de BM25 en la fusión híbrida
        (por defecto PESO_LEXICO; la búsqueda semántica pesa siempre 1.0).
        `penalizacion_ruido` multiplica la puntuación de los fragmentos marcados como de
        bajo valor, en todos los modos (por defecto PENALIZACION_RUIDO; 1.0 = desactivada).
        """
        peso_lexico = PESO_LEXICO if peso_lexico is None else peso_lexico
        penalizacion = PENALIZACION_RUIDO if penalizacion_ruido is None else penalizacion_ruido
        if modo not in MODOS_BUSQUEDA:
            raise ValueError(f"Modo de búsqueda desconocido: {modo!r}")
        if self.vacio or self.vector_store is None or self._bm25 is None:
            return []

        densos: list[Document] = []
        lexicos: list[Document] = []
        if modo in ("hibrido", "semantico"):
            densos = self.vector_store.similarity_search(
                pregunta, k=min(fetch_k, len(self.fragmentos))
            )
        if modo in ("hibrido", "lexico"):
            lexicos = self._bm25.invoke(pregunta)

        # En un solo modo la otra lista está vacía; el peso de BM25 solo importa al fusionar.
        peso_bm25 = peso_lexico if modo == "hibrido" else 1.0
        puntuaciones: dict[tuple, float] = defaultdict(float)
        docs: dict[tuple, Document] = {}
        for lista, peso in ((densos, 1.0), (lexicos, peso_bm25)):
            for posicion, doc in enumerate(lista):
                clave = _clave(doc)
                docs[clave] = doc
                puntuaciones[clave] += peso / (RRF_K + posicion + 1)

        for clave, doc in docs.items():
            if doc.metadata.get("ruido"):
                puntuaciones[clave] *= penalizacion

        mejores = sorted(puntuaciones, key=puntuaciones.get, reverse=True)[:k]
        return [docs[c] for c in mejores]


# --- Utilidades de presentación del contexto -----------------------------------
def construir_contexto(documentos: list[Document]) -> str:
    """Delimita cada fragmento con su procedencia para que el LLM pueda citarla."""
    return "\n\n".join(
        f'<fragmento fuente="{d.metadata.get("source", "desconocido")}" '
        f'pagina="{d.metadata.get("page", "?")}">\n{d.page_content}\n</fragmento>'
        for d in documentos
    )


def formatear_fuentes(documentos: list[Document]) -> str:
    """Fuentes generadas desde los metadatos reales (no inventadas por el LLM)."""
    paginas: dict[str, set] = defaultdict(set)
    for d in documentos:
        paginas[d.metadata.get("source", "desconocido")].add(d.metadata.get("page", "?"))

    partes = []
    for fuente, pags in sorted(paginas.items()):
        lista = ", ".join(str(p) for p in sorted(pags, key=str))
        partes.append(f"{fuente} (p. {lista})")
    return "**Fuentes:** " + "; ".join(partes) if partes else ""