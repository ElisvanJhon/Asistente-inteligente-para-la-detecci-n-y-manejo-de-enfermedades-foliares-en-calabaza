# Chatbot agrícola RAG - Enfermedades de hojas de calabaza

## 1. Estructura del proyecto
```
chatbot_calabaza/
├── models/
│   └── best.pt              ← copia aquí tu modelo YOLOv8n-cls entrenado en Colab
├── knowledge_base/          ← documentos ya incluidos (uno por enfermedad)
├── chroma_db/                ← se genera automáticamente al correr build_kb.py
├── build_kb.py
├── classifier.py
├── retriever.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## 2. Instalación (en VS Code, terminal integrada)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 3. Configurar tu API key (gratis, sin tarjeta)

Ve a **https://aistudio.google.com/apikey**, inicia sesión con tu cuenta de
Google y crea una API key. No pide tarjeta.

Copia `.env.example` a un archivo nuevo llamado `.env` y pega tu key ahí:

```
GEMINI_API_KEY=tu-api-key-real
```

Los embeddings (para la búsqueda del RAG) usan un modelo local
(`sentence-transformers`) que corre en tu PC — no necesita ninguna key.

## 4. Copiar tu modelo entrenado

Copia el archivo `best.pt` que bajaste de Colab a:
```
models/best.pt
```

## 5. Construir la base de conocimiento (solo la primera vez)

```bash
python build_kb.py
```

Esto lee los documentos en `knowledge_base/`, los divide en chunks, genera
embeddings con un modelo local (gratis) y los guarda en una base vectorial
local (Chroma) en la carpeta `chroma_db/`. La primera vez descarga el
modelo de embeddings (~400MB), luego queda en caché.

## 6. Correr el chatbot

```bash
python app.py
```

Esto abre una interfaz web local (Gradio) donde puedes:
1. Subir una foto de una hoja de calabaza → obtienes el diagnóstico
2. Hacer preguntas sobre esa enfermedad → el chatbot responde usando el
   diagnóstico + la base de conocimiento (RAG)

## 7. Ablation study: RAG vs sin RAG (para el paper)

Corre el mismo set de preguntas para las 5 clases, generando la respuesta
CON contexto recuperado y SIN contexto (baseline), y guarda todo en un CSV:

```bash
python ablation_study.py
```

Esto genera `ablation_results.csv` con columnas: `clase`, `confianza`,
`pregunta`, `respuesta_con_rag`, `respuesta_sin_rag`. Úsalo como input
para el siguiente paso: evaluación con LLM-as-judge.

No usa el clasificador de imágenes — evalúa el módulo de texto (RAG)
directamente con diagnósticos simulados, ya que el objetivo es aislar el
efecto del retrieval, no la parte de visión.

## Notas importantes

- **Los nombres de clase en `build_kb.py`** (`FILENAME_TO_CLASS`) deben
  coincidir EXACTAMENTE con los nombres de carpeta que usaste al entrenar
  YOLO (`Bacterial_Leaf_Spot`, `Downy_Mildew`, etc.). Si tu modelo detecta
  clases con otro nombre, ajústalo ahí.
- **Los documentos de `knowledge_base/`** son un punto de partida con
  información agronómica general. Para el paper, te recomiendo
  complementarlos o contrastarlos con fuentes académicas/técnicas
  específicas de tu región (BARI, FAO, extensión agrícola local) y citar
  esas fuentes en la metodología.
- **Costo**: $0. Los embeddings corren localmente en tu PC y Gemini 2.5
  Flash tiene tier gratis (1,500 solicitudes/día, sin tarjeta, sin
  vencimiento). Para más detalles de límites revisa
  https://ai.google.dev/gemini-api/docs/pricing
- **Para el paper**: documenta que usas Gemini 2.5 Flash como LLM
  generador y `paraphrase-multilingual-MiniLM-L12-v2` como modelo de
  embeddings — son datos que necesitas para la sección de metodología.
