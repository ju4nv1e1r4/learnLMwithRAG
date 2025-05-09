import logging
import traceback

import dotenv
import gradio as gr

from src.rag.rag import RetrievalAugmentedGeneration
from utils.config import LoadEnvVars

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
dotenv.load_dotenv()

rag_instance = None
messages_history = []
SHORT_TERM_MEMORY = 15


def get_context(messages):
    latest_messages = messages[-SHORT_TERM_MEMORY:] if messages else []
    context = ""
    for message in latest_messages:
        role = "Usuário" if message["role"] == "user" else "Assistente"
        context += f"{role}: {message['content']}\n"
    return context


def start_rag(pdf_path):
    global rag_instance

    if pdf_path is None:
        return "Erro: Nenhum arquivo foi selecionado. Por favor, selecione um PDF."

    api_key = LoadEnvVars("GOOGLE_API_KEY")
    key = api_key.get_key()

    try:
        rag_instance = RetrievalAugmentedGeneration(key, pdf_path)
        rag_instance.prepare_docs()
        _ = rag_instance.retriever()
        return "PDF carregado com sucesso! Você pode começar a fazer perguntas."
    except Exception as e:
        logging.error(f"Erro ao inicializar RAG: {e}")
        logging.error(traceback.format_exc())
        return f"Erro ao carregar o PDF: {str(e)}"


def generate_response(message, history):
    global rag_instance, messages_history

    messages_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for i, (msg, _) in enumerate(history)
    ] + [{"role": "user", "content": message}]

    if rag_instance is None:
        return "Por favor, carregue um PDF primeiro utilizando o botão acima."

    try:
        context = get_context(messages_history)

        input_dict = {"context": context, "question": message}

        rag = rag_instance.chain()
        response = rag.invoke(input_dict)

        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("answer", str(response))
        else:
            return str(response)
    except Exception as e:
        logging.error(f"Erro ao processar pergunta: {e}")
        logging.error(traceback.format_exc())
        return f"Ocorreu um erro ao consultar a LLM: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("## LearnLM - Assistente RAG com PDF")

    with gr.Row():
        with gr.Column(scale=4):
            file_upload = gr.File(label="Envie seu PDF", file_types=[".pdf"])
        with gr.Column(scale=1):
            upload_btn = gr.Button("Carregar PDF", variant="primary")

    upload_status = gr.Textbox(label="Status do carregamento", interactive=False)

    chat = gr.ChatInterface(
        fn=generate_response,
        examples=[
            "Qual é o tema principal deste documento?",
            "Poderia resumir o conteúdo do PDF?",
        ],
        title="LearnLM Chat",
        description="Faça perguntas sobre o documento carregado",
    )

    upload_btn.click(fn=start_rag, inputs=file_upload, outputs=upload_status)

if __name__ == "__main__":
    demo.launch()
