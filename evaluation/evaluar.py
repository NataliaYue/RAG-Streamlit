"""Evalúa la recuperación: ¿aparece la página correcta entre los k fragmentos devueltos?

Compara la búsqueda semántica, la léxica (BM25) y la híbrida. Mide solo la recuperación,
no la calidad de la respuesta del LLM.

Uso (desde la raíz del proyecto):

    # Indexa los PDFs de una carpeta en memoria y evalúa
    python -m evaluation.evaluar --pdfs manuales --preguntas evaluation/preguntas.json

    # Usa el índice guardado por la app (carpeta data/index)
    python -m evaluation.evaluar --preguntas evaluation/preguntas.json --verbose

Formato de preguntas.json (ver preguntas.ejemplo.json):

    [{"pregunta": "¿Qué significa E-42?", "fuente": "manual.pdf", "paginas": [12]}]

`paginas` admite un número o una lista (si la respuesta puede estar en varias páginas).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.document_processor import (
    FETCH_K,
    MODOS_BUSQUEDA,
    IndiceDocumental,
    parsear_rangos,
)


class ArchivoLocal:
    """Imita el UploadedFile de Streamlit (solo .name y .getvalue())."""

    def __init__(self, ruta: Path):
        self.name = ruta.name
        self._ruta = ruta

    def getvalue(self) -> bytes:
        return self._ruta.read_bytes()


def cargar_casos(ruta: Path | str) -> list[dict]:
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if not isinstance(datos, list) or not datos:
        raise ValueError("El archivo de preguntas debe ser una lista no vacía.")

    casos = []
    for i, caso in enumerate(datos, start=1):
        if not isinstance(caso, dict):
            raise ValueError(f"Caso {i}: debe ser un objeto JSON.")
        faltan = {"pregunta", "fuente", "paginas"} - set(caso)
        if faltan:
            raise ValueError(f"Caso {i}: faltan las claves {sorted(faltan)}.")
        paginas = caso["paginas"] if isinstance(caso["paginas"], list) else [caso["paginas"]]
        casos.append(
            {
                "pregunta": caso["pregunta"],
                "fuente": caso["fuente"],
                "paginas": {int(p) for p in paginas},
            }
        )
    return casos


def indexar_directorio(
    directorio: Path, excluir_paginas: set[int] | None = None
) -> IndiceDocumental:
    pdfs = sorted(directorio.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No hay PDFs en {directorio}")
    indice = IndiceDocumental()
    for aviso in indice.agregar_archivos(
        [ArchivoLocal(p) for p in pdfs], excluir_paginas=excluir_paginas
    ):
        print(f"Aviso: {aviso}", file=sys.stderr)
    return indice


def traducir_casos(casos: list[dict], idioma_destino: str) -> list[dict]:
    """Pasa cada pregunta por el mismo reformulador de la app (flujo real, con el LLM)."""
    from backend.llm_chain import reformular_consulta

    traducidos = []
    for caso in casos:
        pregunta, ok = reformular_consulta("", caso["pregunta"], idioma_destino=idioma_destino)
        traducidos.append(
            {**caso, "original": caso["pregunta"], "pregunta": pregunta, "traduccion_ok": ok}
        )
    return traducidos


def posicion_acierto(documentos, caso: dict) -> int | None:
    """Posición (1 = primero) del primer fragmento correcto, o None si no aparece."""
    for posicion, doc in enumerate(documentos, start=1):
        if doc.metadata.get("source") == caso["fuente"] and doc.metadata.get("page") in caso["paginas"]:
            return posicion
    return None


def evaluar_modo(
    indice,
    casos: list[dict],
    modo: str,
    k: int,
    peso_lexico: float | None = None,
    etiqueta: str | None = None,
    penalizacion: float | None = None,
) -> dict:
    cortes = sorted({n for n in (1, 3, k) if n <= k})
    aciertos = {n: 0 for n in cortes}
    suma_rr = 0.0
    fallos = []
    extra = {}
    if peso_lexico is not None:
        extra["peso_lexico"] = peso_lexico
    if penalizacion is not None:
        extra["penalizacion_ruido"] = penalizacion

    for caso in casos:
        documentos = indice.buscar(
            caso["pregunta"], k=k, fetch_k=max(k, FETCH_K), modo=modo, **extra
        )
        posicion = posicion_acierto(documentos, caso)
        if posicion is None:
            fallos.append(
                {
                    "pregunta": caso["pregunta"],
                    "esperado": f"{caso['fuente']} p.{sorted(caso['paginas'])}",
                    "recuperado": [
                        f"{d.metadata.get('source')} p.{d.metadata.get('page')}"
                        for d in documentos
                    ],
                }
            )
            continue
        suma_rr += 1.0 / posicion
        for n in cortes:
            if posicion <= n:
                aciertos[n] += 1

    total = len(casos)
    return {
        "modo": etiqueta or modo,
        "total": total,
        "hit": {n: aciertos[n] / total for n in cortes},
        "mrr": suma_rr / total,
        "fallos": fallos,
    }


def imprimir_informe(resultados: list[dict], verbose: bool = False) -> None:
    cortes = list(resultados[0]["hit"])
    cabecera = f"{'Modo':<12}" + "".join(f"{'Hit@' + str(n):>8}" for n in cortes) + f"{'MRR':>8}"
    print(cabecera)
    print("-" * len(cabecera))
    for r in resultados:
        fila = f"{r['modo']:<12}" + "".join(f"{r['hit'][n]:>8.2f}" for n in cortes)
        print(fila + f"{r['mrr']:>8.2f}")

    if verbose:
        for r in resultados:
            if not r["fallos"]:
                continue
            print(f"\nFallos en modo {r['modo']} ({len(r['fallos'])}/{r['total']}):")
            for fallo in r["fallos"]:
                print(f"  - {fallo['pregunta']}")
                print(f"      esperado:   {fallo['esperado']}")
                print(f"      recuperado: {', '.join(fallo['recuperado']) or '(nada)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa la recuperación de ManualsQA.")
    parser.add_argument("--preguntas", required=True, help="JSON con las preguntas de prueba")
    parser.add_argument("--pdfs", help="Carpeta con PDFs a indexar (si no, usa el índice guardado)")
    parser.add_argument("--k", type=int, default=5, help="Fragmentos a recuperar (por defecto 5)")
    parser.add_argument(
        "--modos", nargs="+", choices=MODOS_BUSQUEDA, default=list(MODOS_BUSQUEDA)
    )
    parser.add_argument(
        "--excluir-paginas",
        help='Páginas a no indexar con --pdfs, p. ej. "125-148" (apéndices regulatorios)',
    )
    parser.add_argument(
        "--pesos-lexico",
        nargs="+",
        type=float,
        help="Pesos de BM25 a comparar en el modo híbrido, p. ej. 0 0.1 0.3 1",
    )
    parser.add_argument(
        "--penalizacion",
        type=float,
        help="Factor para fragmentos de bajo valor (apéndices, índices, otro idioma); "
        "1 = desactivado. Por defecto, el de la configuración (PENALIZACION_RUIDO)",
    )
    parser.add_argument(
        "--traducir",
        action="store_true",
        help="Traduce cada pregunta al idioma de los manuales con el LLM (necesita "
        "GROQ_API_KEY) y evalúa con la traducción, como hace la app",
    )
    parser.add_argument("--verbose", action="store_true", help="Muestra las preguntas falladas")
    parser.add_argument("--salida", help="Guarda los resultados en un archivo JSON")
    args = parser.parse_args(argv)

    try:
        casos = cargar_casos(args.preguntas)
    except (OSError, ValueError) as e:
        print(f"Error en el archivo de preguntas: {e}", file=sys.stderr)
        return 1

    try:
        excluir = parsear_rangos(args.excluir_paginas)
        if excluir and not args.pdfs:
            print(
                "Aviso: --excluir-paginas solo se aplica con --pdfs; el índice guardado "
                "se usa tal cual (reindexa desde la app para cambiarlo).",
                file=sys.stderr,
            )
        indice = (
            indexar_directorio(Path(args.pdfs), excluir) if args.pdfs else IndiceDocumental.cargar()
        )
    except (OSError, ValueError) as e:
        print(f"Error al preparar el índice: {e}", file=sys.stderr)
        return 1

    if indice.vacio:
        print("El índice está vacío: usa --pdfs o indexa manuales desde la app.", file=sys.stderr)
        return 1

    traducidas = False
    if args.traducir:
        from dotenv import load_dotenv

        load_dotenv()
        from backend.llm_chain import hay_api_key

        if not hay_api_key():
            print("--traducir necesita GROQ_API_KEY (en el entorno o en .env).", file=sys.stderr)
            return 1
        destino = indice.idioma_principal
        if not destino:
            print(
                "No se puede deducir un idioma principal claro del índice, así que no hay "
                "idioma al que traducir.",
                file=sys.stderr,
            )
            return 1
        casos = traducir_casos(casos, destino)
        fallos = sum(not c["traduccion_ok"] for c in casos)
        if fallos == len(casos):
            # Evaluar con las preguntas sin traducir daría un informe que parece de la
            # traducción y no lo es: mejor parar.
            print(
                f"La traducción falló en las {fallos} preguntas (mira el aviso del log: "
                "lo habitual es un modelo que tu cuenta de Groq no tiene). Revisa "
                "GROQ_REFORMULATOR_MODEL o GROQ_MODEL y vuelve a ejecutar.",
                file=sys.stderr,
            )
            return 1
        if fallos:
            print(
                f"Aviso: {fallos} de {len(casos)} preguntas no se pudieron traducir y se "
                "evalúan en su idioma original.",
                file=sys.stderr,
            )
        traducidas = True

    print(
        f"{len(casos)} preguntas · {indice.n_documentos} documentos · "
        f"{indice.n_fragmentos} fragmentos ({indice.n_ruido} de bajo valor detectados) · "
        f"k={args.k}\n"
    )
    resultados = [
        evaluar_modo(indice, casos, modo, args.k, penalizacion=args.penalizacion)
        for modo in args.modos
    ]
    if args.pesos_lexico:
        resultados += [
            evaluar_modo(
                indice,
                casos,
                "hibrido",
                args.k,
                peso_lexico=peso,
                etiqueta=f"hib w={peso:g}",
                penalizacion=args.penalizacion,
            )
            for peso in args.pesos_lexico
        ]
    imprimir_informe(resultados, verbose=args.verbose)

    if traducidas and args.verbose:
        print("\nTraducciones usadas para buscar:")
        for caso in casos:
            print(f"  - {caso['original']}\n      -> {caso['pregunta']}")

    if args.salida:
        Path(args.salida).write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())