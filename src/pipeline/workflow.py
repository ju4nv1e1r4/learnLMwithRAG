import traceback

import dotenv
import streamlit as st

from src.rag.rag import RetrievalAugmentedGeneration
from utils.config import LoadEnvVars

dotenv.load_dotenv()


class RunPipeline:
    def __init__(self, context_control: int = 15):
        self.rag_instance = None
        self.SHORT_TERM_MEMORY = context_control

    def get_context(self):
        latest_messages = st.session_state.get("messages", [])[
            -self.SHORT_TERM_MEMORY :
        ]
        context = ""
        for message in latest_messages:
            role = "Usuário" if message["role"] == "user" else "Assistente"
            context += f"{role}: {message['content']}\n"
        return context

    def start_rag(self, pdf_path, temperature=0.7, top_k=0.0, top_p=0.0):
        api_key = LoadEnvVars("GOOGLE_API_KEY")
        key = api_key.get_key()
        try:
            self.rag_instance = RetrievalAugmentedGeneration(
                key, pdf_path, temperature, top_k, top_p
            )
            self.rag_instance.prepare_docs()
            _ = self.rag_instance.retriever()
            _ = self.rag_instance.retriever()
            return True
        except Exception as e:
            st.error(f"Error starting RAG: {e}")
            st.error(traceback.format_exc())
            return False

    def generate(self, user_input, with_debug_mode=False):
        if self.rag_instance is None:
            return "Please, upload your PDF before starting the conversation."
        try:
            context = self.get_context()

            input_dict = {"context": context, "question": user_input}

            rag = self.rag_instance.chain()

            response = rag.invoke(input_dict)

            if with_debug_mode:
                # DEBUG: Mostrar o que estamos enviando
                st.write("Enviando para a cadeia RAG:")
                st.write(input_dict)
                # DEBUG: Mostrar a resposta bruta
                st.write("Resposta bruta da cadeia RAG:")
                st.write(response)
            else:
                pass

            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                return response.get("answer", str(response))
            else:
                return str(response)

        except Exception as e:
            st.error(traceback.format_exc())
            return f"An error occurred while consulting the LLM: {e}"
