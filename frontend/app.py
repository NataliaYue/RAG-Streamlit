import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # antes de importar el backend

import streamlit as st

# Permite `streamlit run frontend/app.py` desde la raíz sin instalar el paquete.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.document_processor import (  # noqa: E402
    INDEX_DIR,
    PESO_LEXICO_SIN_TRADUCCION,
    IndiceDocumental,
    construir_contexto,
    formatear_fuentes,
    parsear_rangos,
)
from backend.llm_chain import (  # noqa: E402
    SIN_RESPUESTA,
    formatear_historial,
    hay_api_key,
    inicializar_cadena_rag,
    reformular_consulta,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVATAR_USUARIO = "🧑‍💻"
AVATAR_ASISTENTE = "⚡"


# --- Configuración de página -----------------------------------------------------
def configurar_pagina():
    st.set_page_config(page_title="ManualsQA", page_icon="⚡", layout="centered")

    # Colores heredados del tema de Streamlit: funciona en modo claro y oscuro.
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {background: transparent;}

            .block-container { padding-top: 3rem; max-width: 48rem; }

            h1 { font-weight: 800; font-size: 2.5rem; letter-spacing: -0.05em; margin-bottom: 0.2rem; }
            .subtitle { opacity: 0.65; font-size: 1.1rem; margin-bottom: 2.5rem; }

            .stChatMessage {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }

            [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.25); }

            .stButton>button { border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Estado ------------------------------------------------------------------------
def inicializar_estado():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indice" not in st.session_state:
        try:
            st.session_state.indice = IndiceDocumental.cargar(INDEX_DIR)
        except Exception as e:
            logger.exception("No se pudo cargar el índice guardado")
            st.session_state.indice = IndiceDocumental()
            st.session_state.avisos = [
                f"No se pudo cargar el índice guardado ({e}). Tendrás que reindexar."
            ]


def limpiar_contexto():
    st.session_state.messages = []
    st.session_state.indice = IndiceDocumental()
    IndiceDocumental.eliminar(INDEX_DIR)


def exportar_conversacion() -> str:
    bloques = []
    for msg in st.session_state.messages:
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        bloques.append(f"### {rol}\n\n{msg['content']}\n")
        if msg.get("fuentes"):
            bloques.append(f"{msg['fuentes']}\n")
    return "\n".join(bloques)


# --- Sidebar -------------------------------------------------------------------------
def construir_sidebar():
    indice: IndiceDocumental = st.session_state.indice

    with st.sidebar:
        st.markdown("### Workspace")
        st.write("---")

        col1, col2 = st.columns(2)
        col1.metric("Documentos", indice.n_documentos)
        col2.metric("Fragmentos", indice.n_fragmentos)

        if indice.n_ruido:
            st.caption(
                f"{indice.n_ruido} fragmentos de bajo valor (índices, apéndices, otros idiomas) "
                "se muestran con menos prioridad."
            )

        if indice.n_documentos:
            with st.expander("Documentos indexados"):
                for nombre in indice.nombres:
                    st.write(f"📄 {nombre}")

        for aviso in st.session_state.pop("avisos", []):
            st.warning(aviso)

        st.write("---")

        archivos = st.file_uploader(
            "Añadir manuales",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        with st.expander("Opciones avanzadas"):
            excluir_texto = st.text_input(
                "Excluir páginas (opcional)",
                placeholder="ej. 125-148",
                help=(
                    "Normalmente no hace falta: los apéndices, índices y textos en otros "
                    "idiomas se detectan solos y se rebajan en los resultados. Úsalo solo para "
                    "dejar fuera páginas concretas (números del visor de PDF)."
                ),
            )

        if archivos and st.button("Indexar conocimiento", use_container_width=True):
            try:
                excluir = parsear_rangos(excluir_texto)
            except ValueError as e:
                st.error(str(e))
                return
            fragmentos_antes = indice.n_fragmentos
            with st.spinner("Procesando y vectorizando..."):
                try:
                    avisos = indice.agregar_archivos(archivos, excluir_paginas=excluir)
                except Exception as e:
                    logger.exception("Error indexando")
                    st.error(f"Error al indexar: {e}")
                    return
                if indice.n_fragmentos != fragmentos_antes:
                    try:
                        indice.guardar(INDEX_DIR)
                    except Exception as e:
                        logger.exception("Error guardando el índice")
                        avisos.append(
                            f"El índice funciona, pero no se pudo guardar en disco: {e}"
                        )
            st.session_state.avisos = avisos
            st.rerun()  # refresca métricas y lista de documentos

        if not indice.vacio:
            st.button(
                "Purgar base de datos",
                on_click=limpiar_contexto,
                use_container_width=True,
            )

        if st.session_state.messages:
            st.download_button(
                "Descargar conversación",
                data=exportar_conversacion(),
                file_name="conversacion.md",
                mime="text/markdown",
                use_container_width=True,
            )


# --- Chat ------------------------------------------------------------------------------
def mostrar_extras(mensaje: dict):
    """Fuentes (desde metadatos) y fragmentos usados, para poder verificar la respuesta."""
    if mensaje.get("fuentes"):
        st.markdown(mensaje["fuentes"])
    if mensaje.get("fragmentos"):
        with st.expander("Fragmentos consultados"):
            for frag in mensaje["fragmentos"]:
                st.caption(f"{frag['fuente']} · página {frag['pagina']}")
                st.text(frag["texto"][:600])


def manejar_input_usuario():
    indice: IndiceDocumental = st.session_state.indice

    if not st.session_state.messages and indice.vacio:
        st.info("💡 Sube un documento en el panel lateral para activar el motor de búsqueda.")

    for msg in st.session_state.messages:
        avatar = AVATAR_USUARIO if msg["role"] == "user" else AVATAR_ASISTENTE
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                mostrar_extras(msg)

    prompt = st.chat_input("Consulta el manual (ej. ¿Qué significa la luz roja parpadeante?)")
    if not prompt:
        return

    if indice.vacio:
        st.error("Indexa al menos un documento antes de consultar.")
        return

    # El historial se calcula ANTES de añadir la pregunta actual.
    historial = formatear_historial(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
        try:
            # A) Reformular y traducir al idioma del manual (si falla, usa la original)
            idioma_manual = indice.idioma_principal
            pregunta_busqueda, traducido_ok = reformular_consulta(
                historial, prompt, idioma_destino=idioma_manual
            )
            # Si se esperaba traducir y no se pudo, BM25 pesa poco: en otro idioma no sirve.
            peso_lexico = None
            if idioma_manual and not traducido_ok:
                peso_lexico = PESO_LEXICO_SIN_TRADUCCION
                st.caption(
                    "⚠️ No se pudo traducir la consulta al idioma del manual; "
                    "se busca con la pregunta original, con menos precisión."
                )
            elif pregunta_busqueda != prompt:
                st.caption(f"🔍 Buscando en el manual: *{pregunta_busqueda}*")

            # B) Recuperación híbrida con la pregunta reformulada
            documentos = indice.buscar(pregunta_busqueda, peso_lexico=peso_lexico)

            # C) Generación en streaming con la pregunta original
            respuesta = st.write_stream(
                inicializar_cadena_rag().stream(
                    {
                        "input": prompt,
                        "chat_history": historial,
                        "context": construir_contexto(documentos),
                    }
                )
            )
        except Exception as e:
            logger.exception("Error generando la respuesta")
            st.error(f"No se pudo generar la respuesta: {e}")
            st.session_state.messages.pop()  # evita dejar una pregunta sin respuesta
            return

        mensaje = {"role": "assistant", "content": respuesta}
        if not respuesta.strip().startswith(SIN_RESPUESTA):
            mensaje["fuentes"] = formatear_fuentes(documentos)
            mensaje["fragmentos"] = [
                {
                    "fuente": d.metadata.get("source", "desconocido"),
                    "pagina": d.metadata.get("page", "?"),
                    "texto": d.page_content,
                }
                for d in documentos
            ]
        mostrar_extras(mensaje)
        st.session_state.messages.append(mensaje)


# --- Main ------------------------------------------------------------------------------
def main():
    configurar_pagina()
    inicializar_estado()

    st.markdown("<h1>ManualsQA</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Búsqueda semántica de precisión sobre documentación técnica.</div>',
        unsafe_allow_html=True,
    )

    if not hay_api_key():
        st.error("Falta `GROQ_API_KEY`. Defínela en el archivo `.env` o como variable de entorno.")
        st.stop()

    construir_sidebar()
    manejar_input_usuario()


if __name__ == "__main__":
    main()