# Chatbot agrícola RAG - Enfermedades de hojas de calabaza

## 1. Estructura del proyecto
```
chatbot_calabaza/
├── models/
│   └── best.pt              ← copia aqui el modelo best.pt de la carpeta modelo
├── knowledge_base/          
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




