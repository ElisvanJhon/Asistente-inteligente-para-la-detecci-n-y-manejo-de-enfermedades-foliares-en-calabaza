"""
Ablation study: RAG vs sin RAG.

Corre el mismo set de preguntas para cada clase de enfermedad, generando
la respuesta CON contexto recuperado (RAG) y SIN contexto (baseline), y
guarda todo en un CSV para análisis y para el LLM-as-judge posterior.

Guarda el progreso fila por fila, así que si se corta a mitad de camino
(por rate limit u otro motivo) no pierdes lo ya generado -- solo vuelve
a correr el script y sigue donde se quedó.

Correr con:
    python ablation_study.py

NOTA: el tier gratis de Gemini permite 5 solicitudes/minuto, así que este
script tarda varios minutos en completarse (25 preguntas x 2 versiones,
con ~13s de espera entre cada llamada). Es normal que tarde ~20 minutos.
"""

import csv
from pathlib import Path

from rag_core import responder_con_rag, responder_sin_rag

CLASES = {
    "Bacterial_Leaf_Spot": 83.0,
    "Downy_Mildew": 85.0,
    "Healthy_Leaf": 95.0,
    "Mosaic_Disease": 80.0,
    "Powdery_Mildew": 78.0,
}

PREGUNTAS = [
    "¿Qué tratamiento le doy?",
]

OUTPUT_CSV = Path("ablation_results.csv")
FIELDNAMES = ["clase", "confianza", "pregunta", "respuesta_con_rag", "respuesta_sin_rag"]


def cargar_ya_procesadas() -> set:
    """Lee el CSV existente (si lo hay) para saber qué pares clase+pregunta ya se hicieron."""
    if not OUTPUT_CSV.exists():
        return set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {(row["clase"], row["pregunta"]) for row in reader}


def guardar_fila(fila: dict):
    existe = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not existe:
            writer.writeheader()
        writer.writerow(fila)


def main():
    ya_procesadas = cargar_ya_procesadas()
    if ya_procesadas:
        print(f"Reanudando: {len(ya_procesadas)} pares ya procesados anteriormente.\n")

    total = len(CLASES) * len(PREGUNTAS)
    contador = 0

    for clase, confianza in CLASES.items():
        for pregunta in PREGUNTAS:
            contador += 1
            if (clase, pregunta) in ya_procesadas:
                print(f"[{contador}/{total}] {clase} -> \"{pregunta}\" (ya hecho, saltando)")
                continue

            print(f"[{contador}/{total}] {clase} -> \"{pregunta}\"")

            respuesta_con_rag = responder_con_rag(pregunta, clase, confianza)
            respuesta_sin_rag = responder_sin_rag(pregunta, clase, confianza)

            guardar_fila({
                "clase": clase,
                "confianza": confianza,
                "pregunta": pregunta,
                "respuesta_con_rag": respuesta_con_rag,
                "respuesta_sin_rag": respuesta_sin_rag,
            })

    print(f"\nListo. Resultados completos en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
