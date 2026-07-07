"""
Chatbot agrícola con RAG para diagnóstico de enfermedades en hojas de calabaza.

Generador: Gemini 2.5 Flash (gratis, sin tarjeta -> aistudio.google.com/apikey)
Embeddings: modelo local (sentence-transformers), gratis, sin API.

Correr con:
    python app.py
"""

import gradio as gr

from classifier import LeafClassifier
from rag_core import responder_con_rag

classifier = LeafClassifier(weights_path="models/best.pt")


def diagnosticar(imagen_path: str) -> dict:
    return classifier.predict(imagen_path)


def responder(mensaje: str, diagnostico: dict | None) -> str:
    if diagnostico is None:
        return "Primero sube una foto de la hoja para poder darte un diagnóstico y recomendaciones."
    return responder_con_rag(
        mensaje=mensaje,
        clase=diagnostico["clase"],
        confianza=diagnostico["confianza"] * 100,
    )


# ---------------- Interfaz Gradio ----------------

with gr.Blocks(title="Chatbot agrícola - Calabaza") as demo:
    gr.Markdown("## 🎃 Chatbot agrícola: diagnóstico de enfermedades en hojas de calabaza")

    diagnostico_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            imagen_input = gr.Image(type="filepath", label="Foto de la hoja")
            btn_diagnosticar = gr.Button("Diagnosticar", variant="primary")
            resultado_diagnostico = gr.Markdown()

        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Asistente", type="messages", height=420)
            mensaje_input = gr.Textbox(label="Tu pregunta", placeholder="¿Qué tratamiento le doy?")
            btn_enviar = gr.Button("Enviar")

    def on_diagnosticar(imagen_path):
        if imagen_path is None:
            return "Sube una imagen primero.", None
        diag = diagnosticar(imagen_path)
        texto = (
            f"**Diagnóstico:** {diag['clase']}  \n"
            f"**Confianza:** {diag['confianza'] * 100:.1f}%"
        )
        return texto, diag

    btn_diagnosticar.click(
        on_diagnosticar,
        inputs=[imagen_input],
        outputs=[resultado_diagnostico, diagnostico_state],
    )

    def on_enviar(mensaje, historial, diag):
        if not mensaje:
            return historial, ""
        respuesta = responder(mensaje, diag)
        historial = historial + [
            {"role": "user", "content": mensaje},
            {"role": "assistant", "content": respuesta},
        ]
        return historial, ""

    btn_enviar.click(
        on_enviar,
        inputs=[mensaje_input, chatbot_ui, diagnostico_state],
        outputs=[chatbot_ui, mensaje_input],
    )
    mensaje_input.submit(
        on_enviar,
        inputs=[mensaje_input, chatbot_ui, diagnostico_state],
        outputs=[chatbot_ui, mensaje_input],
    )

if __name__ == "__main__":
    demo.launch()
