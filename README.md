# ManualsQA
 
Asistente de preguntas y respuestas sobre manuales técnicos en PDF. Subes tus manuales, haces preguntas en lenguaje natural y la aplicación responde **solo con lo que dicen los manuales**, indicando de qué documento y página sale cada respuesta.
 
Construido con **Streamlit** (interfaz), **LangChain** (orquestación), **FAISS + BM25** (búsqueda) y **Groq** (modelo de lenguaje).
 
---
 
## Qué es RAG
 
RAG (*Retrieval-Augmented Generation*, generación aumentada por recuperación) combina dos piezas:
 
1. **Recuperación**: buscar en tus documentos los fragmentos más relevantes para la pregunta.
2. **Generación**: pasar esos fragmentos a un modelo de lenguaje con la orden de responder usando únicamente esa información.
Así el modelo no "se inventa" la respuesta a partir de lo que aprendió en su entrenamiento: redacta a partir de tus manuales, y la aplicación puede citar las fuentes.
 
---
 
## Cómo funciona
 
La aplicación tiene dos fases.
 
### 1. Indexar (al subir PDFs)
 
```
Subir PDFs ─► Extraer texto por página ─► Marcar índices y apéndices
           ─► Trocear en fragmentos ─► Detectar idioma
           ─► Guardar en FAISS + BM25 (y en disco)
```
 
### 2. Preguntar
 
```
Escribir la pregunta ─► Reformular y traducir al idioma del manual
                     ─► Búsqueda híbrida (FAISS + BM25)
                     ─► Rebajar el ruido y elegir los 5 mejores fragmentos
                     ─► Generar la respuesta ─► Mostrar respuesta y fuentes
```
 
El índice que se construye en la fase 1 es el mismo en el que se busca en la fase 2. Se guarda en disco, así que los manuales siguen indexados aunque cierres la aplicación.
 
---
 
## Estructura del proyecto
 
```
.
├── backend/
│   ├── document_processor.py   # Leer PDFs, trocear, indexar, guardar y buscar
│   ├── limpieza.py             # Detectar ruido e idioma; tokenizador para BM25
│   └── llm_chain.py            # Llamadas al modelo: reformular y responder
├── frontend/
│   └── app.py                  # Interfaz Streamlit; orquesta todo lo demás
├── data/index/                 # Índice guardado (se genera solo, no se sube)
├── .env.example                # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```
 
| Fichero | Cuándo actúa | Qué hace |
|---|---|---|
| `frontend/app.py` | Siempre | Recibe lo que haces y muestra el resultado |
| `backend/document_processor.py` | Indexar y preguntar | El trabajo pesado: leer, trocear, guardar y buscar |
| `backend/limpieza.py` | Al indexar | Etiqueta cada fragmento (ruido, idioma) para priorizar en la búsqueda |
| `backend/llm_chain.py` | Al preguntar | Habla con el modelo dos veces: reescribir la pregunta y redactar la respuesta |
 
---
 
## Instalación y uso
 
Requiere Python 3.10 o superior y una API key de [Groq](https://console.groq.com).
 
```bash
git clone <url-de-tu-repo>
cd <carpeta-del-repo>
 
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
cp .env.example .env             # Y pon tu GROQ_API_KEY dentro
streamlit run frontend/app.py
```
 
La primera vez tarda un poco más: descarga el modelo de embeddings (unos cientos de MB).
 
---
 
## Explicación del código
 
### `frontend/app.py`: la interfaz y el director de orquesta
 
**Cómo funciona Streamlit.** Streamlit vuelve a ejecutar el script entero de arriba abajo cada vez que el usuario interactúa (pulsa un botón, escribe una pregunta…). Lo que deba sobrevivir entre ejecuciones se guarda en `st.session_state`. Aquí se guardan la conversación (`messages`) y el índice (`indice`).
 
**Arranque.** `load_dotenv()` se llama antes de importar el backend porque esos módulos leen las variables de entorno en cuanto se importan.
 
**Funciones principales:**
 
- `inicializar_estado()`: carga el índice guardado en disco la primera vez. Si no puede, empieza con uno vacío y muestra un aviso.
- `construir_sidebar()`: métricas, subida de PDFs y botón de indexar. Al indexar llama a `indice.agregar_archivos(...)`, guarda en disco y refresca la página con `st.rerun()`.
- `manejar_input_usuario()`: el flujo de una pregunta.
  1. Repinta la conversación (Streamlit lo borra todo en cada ejecución).
  2. Calcula el historial **antes** de añadir la pregunta nueva.
  3. **Reformula** la pregunta: la hace autónoma y la traduce al idioma del manual. Si la traducción falla, desactiva la búsqueda por palabras (BM25), porque buscar palabras en otro idioma no sirve.
  4. **Busca** los 5 fragmentos más relevantes con la pregunta reformulada.
  5. **Genera** la respuesta en streaming con la pregunta **original**, para que responda en el idioma del usuario.
  6. Añade las fuentes solo si el modelo encontró la respuesta.
- `mostrar_extras()`: muestra las fuentes y los fragmentos exactos usados, para poder verificar la respuesta.
> Idea clave: se **busca** con la pregunta reformulada (para encontrar mejor) pero se **responde** a la original (para que la respuesta encaje con lo que se preguntó).
 
### `backend/document_processor.py`: indexado y búsqueda
 
<!-- Por completar -->
 
### `backend/limpieza.py`: detección de ruido e idioma
 
<!-- Por completar -->
 
### `backend/llm_chain.py`: el modelo de lenguaje
 
<!-- Por completar -->
 
---
 
## Configuración
 
Todas las opciones se pueden cambiar con variables de entorno (en el `.env`).
 
| Variable | Por defecto | Para qué sirve |
|---|---|---|
| `GROQ_API_KEY` | — | Obligatoria. Clave de la API de Groq |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Modelo que redacta las respuestas |
| `GROQ_REFORMULATOR_MODEL` | el mismo | Modelo que reformula y traduce la pregunta |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings para la búsqueda semántica |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Tamaño de los fragmentos y solapamiento entre ellos |
| `TOP_K` | `5` | Fragmentos que llegan al modelo |
| `FETCH_K` | `20` | Candidatos de cada buscador antes de fusionar |
| `PESO_LEXICO` | `1.0` | Peso de BM25 frente a la búsqueda semántica |
| `PENALIZACION_RUIDO` | `0.7` | Factor que rebaja los fragmentos marcados como ruido |
| `INDEX_DIR` | `data/index` | Dónde se guarda el índice |
 
<!-- Ampliar la explicación de cada parámetro al repasar document_processor.py -->