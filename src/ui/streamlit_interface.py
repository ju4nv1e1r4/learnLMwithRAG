from src.rag.rag import RetrievalAugmentedGeneration
from utils.config import LoadEnvVars
import streamlit as st
import dotenv
import traceback

dotenv.load_dotenv()

rag_instance = None
SHORT_TERM_MEMORY = 15

def get_context():
    latest_messages = st.session_state.get("messages", [])[-SHORT_TERM_MEMORY:]
    context = ""
    for message in latest_messages:
        role = "Usuário" if message["role"] == "user" else "Assistente"
        context += f"{role}: {message['content']}\n"
    return context

def start_rag(pdf_path, temperature=0.7, top_k=0.0, top_p=0.0):
    global rag_instance
    api_key = LoadEnvVars("GOOGLE_API_KEY")
    key = api_key.get_key()
    try:
        rag_instance = RetrievalAugmentedGeneration(key, pdf_path, temperature, top_k, top_p)
        rag_instance.prepare_docs()
        _ = rag_instance.retriever()
        return True
    except Exception as e:
        st.error(f"Erro ao inicializar RAG: {e}")
        st.error(traceback.format_exc())
        return False

def generate(user_input):
    global rag_instance
    if rag_instance is None:
        return "Por favor, carregue um PDF primeiro."
    try:
        context = get_context()
        
        input_dict = {
            "context": context,
            "question": user_input
        }
        
        # DEBUG: Mostrar o que estamos enviando
        # st.write("Enviando para a cadeia RAG:")
        # st.write(input_dict)
        
        rag = rag_instance.chain()
        
        response = rag.invoke(input_dict)
        
        # DEBUG: Mostrar a resposta bruta
        # st.write("Resposta bruta da cadeia RAG:")
        # st.write(response)
        
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get('answer', str(response))
        else:
            return str(response)
            
    except Exception as e:
        st.error(traceback.format_exc())
        return f"Ocorreu um erro ao consultar a LLM: {e}"

def main():
    st.markdown("# Instrutor LLM com LearnLM :brain: :teacher:")

    with st.sidebar:
        temperature = st.selectbox(
            "Temperature",
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            index=4,
            placeholder="Defina o parâmetro temperature"
        )

        top_k = st.selectbox(
            "Top K",
            (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100),
            index=4,
            placeholder="Defina o parâmetro top_k"
        )

        top_p = st.selectbox(
            "Top P",
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            index=4,
            placeholder="Defina o parâmetro top_p"
        )
    
    pdf = st.file_uploader("Carregue um PDF", type="pdf")
    if pdf is not None:
        with open("temp.pdf", "wb") as f:
            f.write(pdf.read())
        if start_rag("temp.pdf", temperature, top_k, top_p):
            st.success("PDF carregado. Agora você pode conversar!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input('Vamos começar...'):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.markdown(f'{prompt}')
        
        try:
            with st.chat_message('assistant'):
                with st.spinner("Processando..."):
                    response = generate(prompt)                    
                    st.markdown(f'{response}')
        except Exception as e:
            response = "Desculpe, ocorreu um erro ao processar sua solicitação."
            st.error(f"Erro: {e}")
            st.error(traceback.format_exc())
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        if len(st.session_state.messages) > SHORT_TERM_MEMORY * 2:
            st.session_state.messages = st.session_state.messages[-SHORT_TERM_MEMORY * 2:]

if __name__ == "__main__":
    main()