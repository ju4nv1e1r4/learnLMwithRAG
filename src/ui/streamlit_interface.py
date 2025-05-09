import traceback

import streamlit as st

from src.pipeline.workflow import RunPipeline


def main():
    st.markdown("# LLM Tutor with LearnLM :brain: :teacher:")

    with st.sidebar:
        temperature = st.selectbox(
            "Temperature",
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            index=4,
            placeholder="Choose the temperature",
        )

        top_k = st.selectbox(
            "Top K",
            (
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
                100,
            ),
            index=4,
            placeholder="Tune the parameter top_k",
        )

        top_p = st.selectbox(
            "Top P",
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            index=4,
            placeholder="Define the parameter top_p",
        )

        context_control = st.selectbox(
            "Context Control",
            (10, 15, 20, 25, 30, 35, 40, 45, 50),
            index=4,
            placeholder="Tune the context control of conversation",
        )

    run = RunPipeline(context_control=context_control)
    memory = run.SHORT_TERM_MEMORY

    pdf = st.file_uploader("Upload your PDF here", type="pdf")
    if pdf is not None:
        with open("temp.pdf", "wb") as f:
            f.write(pdf.read())
        if run.start_rag("temp.pdf", temperature, top_k, top_p):
            st.success("PDF uploaded. Now you can start to learn!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Let's learn something together..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"{prompt}")

        try:
            with st.chat_message("assistant"):
                with st.spinner("Tutor is formulating an answer..."):
                    response = run.generate(prompt)
                    st.markdown(f"{response}")
        except Exception as e:
            response = "I'm sorry, an error occurred while processing your question."
            st.error(f"Erro: {e}")
            st.error(traceback.format_exc())

        st.session_state.messages.append({"role": "assistant", "content": response})

        if len(st.session_state.messages) > memory * 2:
            st.session_state.messages = st.session_state.messages[-memory * 2 :]


if __name__ == "__main__":
    main()
