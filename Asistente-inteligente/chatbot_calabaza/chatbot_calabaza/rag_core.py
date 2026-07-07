"""
Lógica compartida del sistema RAG: cliente LLM, retriever, y las dos
funciones de respuesta (con RAG y sin RAG) usadas tanto por app.py como
por ablation_study.py.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from retriever import KnowledgeRetriever

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "No se encontró GEMINI_API_KEY. Copia .env.example a .env y pega tu "
        "API key de https://aistudio.google.com/apikey ahí."
    )

client_llm = genai.Client(api_key=GEMINI_API_KEY)
retriever = KnowledgeRetriever()

# El tier gratis de gemini-2.5-flash permite 5 solicitudes por minuto.
# Dejamos ~13s entre llamadas para no pasarnos (60s / 5 = 12s, +margen).
SEGUNDOS_ENTRE_LLAMADAS = 13
MAX_REINTENTOS = 5

SYSTEM_PROMPT = """Eres un asistente agrícola experto en enfermedades de calabaza.

Instrucciones:
- Responde en lenguaje claro y práctico, no técnico en exceso.
- Basa tu respuesta SOLO en el contexto recuperado y el diagnóstico dados.
- Si el contexto no cubre algo, dilo explícitamente, no inventes.
- Sugiere pasos concretos de manejo/tratamiento cuando aplique.
- Si la confianza del diagnóstico es baja (<70%), menciona que se
  recomienda confirmación visual adicional o una segunda foto."""


def _llamar_llm(user_prompt: str) -> str:
    """Llama al LLM con reintentos automáticos si se topa con rate limit (429)
    o con el servidor saturado (503)."""
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            response = client_llm.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.3,
                },
            )
            time.sleep(SEGUNDOS_ENTRE_LLAMADAS)
            return response.text
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            mensaje_error = str(e)
            if "RESOURCE_EXHAUSTED" in mensaje_error or "429" in mensaje_error:
                espera = 25 * intento  # backoff progresivo: 25s, 50s, 75s...
                print(f"  ⏳ Rate limit alcanzado, esperando {espera}s (intento {intento}/{MAX_REINTENTOS})...")
                time.sleep(espera)
            elif "UNAVAILABLE" in mensaje_error or "503" in mensaje_error:
                espera = 15 * intento  # servidor saturado, backoff más corto
                print(f"  ⏳ Servidor de Gemini saturado (503), esperando {espera}s (intento {intento}/{MAX_REINTENTOS})...")
                time.sleep(espera)
            else:
                raise
    raise RuntimeError("Se agotaron los reintentos. Intenta correr el script más tarde.")


def responder_con_rag(mensaje: str, clase: str, confianza: float) -> str:
    """Pipeline completo: retrieval + generación."""
    chunks = retriever.retrieve(query=mensaje, clase=clase, k=3)
    contexto = "\n\n---\n\n".join(chunks) if chunks else "(sin contexto adicional disponible)"

    user_prompt = f"""DIAGNÓSTICO DEL MODELO DE VISIÓN:
Enfermedad detectada: {clase}
Confianza: {confianza:.1f}%

CONTEXTO RECUPERADO:
{contexto}

PREGUNTA DEL AGRICULTOR:
{mensaje}"""
    return _llamar_llm(user_prompt)


def responder_sin_rag(mensaje: str, clase: str, confianza: float) -> str:
    """Baseline: mismo diagnóstico y pregunta, SIN contexto recuperado."""
    user_prompt = f"""DIAGNÓSTICO DEL MODELO DE VISIÓN:
Enfermedad detectada: {clase}
Confianza: {confianza:.1f}%

PREGUNTA DEL AGRICULTOR:
{mensaje}"""
    return _llamar_llm(user_prompt)
