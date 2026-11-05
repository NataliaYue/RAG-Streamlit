"""Detección automática de contenido de bajo valor para la recuperación.

Sin rangos manuales: marca los fragmentos que suelen colarse en los resultados sin
responder a nada (índices, apéndices legales o regulatorios, texto en un idioma
minoritario dentro del manual). El marcado se guarda en `metadata["ruido"]` y la
búsqueda lo rebaja; nunca se borra nada, así que si el usuario pregunta justo por
eso (p. ej. por la normativa) sigue pudiendo encontrarlo.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from langchain_core.documents import Document

INDICE = "indice"
APENDICE = "apendice"
IDIOMA = "idioma"

# Encabezados que abren una sección final de poco valor para consultas técnicas.
_RE_SECCION_FINAL = re.compile(
    r"^\s*(appendix|appendices|annex|anexo|ap[eé]ndice|appendice|anhang"
    r"|notices|legal notices?|trademarks|regulatory (information|notices)"
    r"|avisos legales|aviso legal|marcas registradas)\b",
    re.IGNORECASE,
)
# Encabezados que cierran esa sección y devuelven al contenido normal.
_RE_CAPITULO = re.compile(
    r"^\s*(chapter|cap[ií]tulo|chapitre|kapitel|capitolo)\s+\d+", re.IGNORECASE
)
_RE_PUNTOS_GUIA = re.compile(r"(?:\.\s?){4,}")  # ". . . . ." típico de los índices
_MIN_PUNTOS_GUIA = 8
_LINEAS_CABECERA = 4

# Palabras funcionales por idioma (suficiente para distinguir idiomas largos).
_PALABRAS = {
    "en": "the of and to in is that for with you your are be this or on as by from can not if it an at will".split(),
    "es": "de la que el en los las del por con para una un se no es como al lo más sus este está su si pero también puede".split(),
    "pt": "de que do da em um uma para com não os as se por mais dos como mas foi ao ele das tem seu sua ou são".split(),
    "fr": "de la le les des et en un une du pour que dans qui par sur pas au avec est ce vous votre".split(),
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch sie ihr".split(),
    "it": "di e il la che in per un è con non una su le si da come dei nel della sono gli".split(),
}
_CONJUNTOS = {idioma: frozenset(p) for idioma, p in _PALABRAS.items()}
_MIN_PALABRAS = 20
_MIN_PUNTUACION = 0.15
_MIN_MARGEN = 0.04

# Un idioma solo se considera "minoritario" si ocupa como mucho esta parte del manual.
MAX_MINORIA = 0.25
# Hacen falta suficientes fragmentos con idioma detectado para fiarse de la mayoría.
MIN_FRAGMENTOS_IDIOMA = 20


NOMBRES_IDIOMA = {
    "en": "inglés",
    "es": "español",
    "pt": "portugués",
    "fr": "francés",
    "de": "alemán",
    "it": "italiano",
}

# Palabras vacías de todos los idiomas reconocidos: BM25 no debe puntuar por "the" o "de".
_PALABRAS_VACIAS = frozenset().union(*_CONJUNTOS.values())
# Conserva códigos y nombres compuestos ("E-42", "M.2", "nano-SIM") como una sola palabra.
_RE_TOKEN = re.compile(r"\w+(?:[-.]\w+)*")


# Guiones "de palabra" (U+2010 guion, U+2011 guion no separable...) -> "-" ASCII, para que
# "nano‑SIM" (lo que escriben los LLM) y "nano-SIM" (lo que trae el manual) coincidan.
# Las rayas de puntuación (U+2012-2015, signo menos) separan palabras. El guion blando
# (U+00AD) es un artefacto de las rupturas de línea de los PDF y se elimina.
_TABLA_UNICODE = {
    **dict.fromkeys((0x2010, 0x2011, 0x2043, 0xFE63, 0xFF0D), "-"),
    **dict.fromkeys((0x2012, 0x2013, 0x2014, 0x2015, 0x2212), " "),
    0x00AD: None,
}


def _normalizar(texto: str) -> str:
    """NFKC (ligaduras "ﬁ" -> "fi", espacios duros, anchos completos) y guiones unificados."""
    return unicodedata.normalize("NFKC", texto).translate(_TABLA_UNICODE)


def tokenizar_bm25(texto: str) -> list[str]:
    """Tokenizador para BM25: minúsculas, Unicode normalizado y sin palabras vacías.

    El de LangChain por defecto solo parte por espacios: distingue mayúsculas, deja la
    puntuación pegada a las palabras y puntúa por "the", "I" o "to", de modo que las
    páginas largas y genéricas ganan casi cualquier consulta. Se aplica igual al texto
    indexado y a la consulta, así que las variantes tipográficas coinciden entre sí.
    """
    return [
        t
        for t in _RE_TOKEN.findall(_normalizar(texto).lower())
        if t not in _PALABRAS_VACIAS and (len(t) > 1 or t.isdigit())
    ]


def _lineas(texto: str) -> list[str]:
    return [l.strip() for l in texto.splitlines() if l.strip()]


def es_pagina_indice(texto: str) -> bool:
    """Un índice de contenidos se reconoce por su densidad de puntos guía."""
    return len(_RE_PUNTOS_GUIA.findall(texto)) >= _MIN_PUNTOS_GUIA


def marcar_paginas(paginas: list[Document]) -> None:
    """Marca índices y secciones finales (apéndices, avisos, marcas) de UN documento.

    `paginas` debe estar en orden. Una sección final empieza en un encabezado tipo
    "Appendix A" y dura hasta el siguiente "Chapter N" o hasta el final.
    """
    en_seccion_final = False
    for doc in paginas:
        # Un índice lista "Appendix…" y "Chapter…" sin ser esas secciones: no cambia el estado.
        if es_pagina_indice(doc.page_content):
            doc.metadata["ruido"] = INDICE
            continue

        cabecera = _lineas(doc.page_content)[:_LINEAS_CABECERA]
        if any(_RE_CAPITULO.match(l) for l in cabecera):
            en_seccion_final = False
        elif any(_RE_SECCION_FINAL.match(l) for l in cabecera):
            en_seccion_final = True

        if en_seccion_final:
            doc.metadata["ruido"] = APENDICE


def detectar_idioma(texto: str) -> str | None:
    """Idioma dominante por palabras funcionales, o None si no es concluyente."""
    palabras = re.findall(r"[^\W\d_]+", texto.lower())
    if len(palabras) < _MIN_PALABRAS:
        return None
    puntuaciones = sorted(
        (
            (sum(p in conjunto for p in palabras) / len(palabras), idioma)
            for idioma, conjunto in _CONJUNTOS.items()
        ),
        reverse=True,
    )
    (mejor, idioma), (segundo, _) = puntuaciones[0], puntuaciones[1]
    if mejor < _MIN_PUNTUACION or mejor - segundo < _MIN_MARGEN:
        return None
    return idioma


def marcar_idiomas(fragmentos: list[Document]) -> None:
    """Marca los fragmentos en un idioma minoritario respecto al resto de su documento.

    Un manual bilingüe (reparto parejo) no se toca: solo se marca lo claramente residual.
    """
    por_fuente: dict[str, list[tuple[Document, str]]] = defaultdict(list)
    for frag in fragmentos:
        idioma = detectar_idioma(frag.page_content)
        if idioma:
            frag.metadata["idioma"] = idioma  # lo usa el índice para saber en qué idioma buscar
            por_fuente[frag.metadata.get("source", "")].append((frag, idioma))

    for items in por_fuente.values():
        if len(items) < MIN_FRAGMENTOS_IDIOMA:
            continue
        conteo = Counter(idioma for _, idioma in items)
        principal = conteo.most_common(1)[0][0]
        for frag, idioma in items:
            es_minoritario = idioma != principal and conteo[idioma] / len(items) <= MAX_MINORIA
            if es_minoritario and not frag.metadata.get("ruido"):
                frag.metadata["ruido"] = IDIOMA