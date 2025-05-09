import logging

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LoadLLM:
    def __init__(self, model_name, api_key, temperature=0.7, top_k=0.0, top_p=0.0):
        self.model_name = model_name
        self.api_key = api_key
        self._llm_instance = None
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def get_llm(self):
        logging.info(f"Temperature: {self.temperature}")
        logging.info(f"Top K: {self.top_k}")
        logging.info(f"Top P: {self.top_p}")
        if self._llm_instance is None:
            self._llm_instance = GoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        return self._llm_instance

    def prompt(self):
        system_prompt = (
            "Seja um tutor amigável e solidário. Oriente o aluno a atingir seus "
            "objetivos, incentivando-o gentilmente a realizar a tarefa caso se desvie."
            "Faça perguntas orientadoras para ajudar seus alunos a darem passos "
            "incrementais em direção à compreensão de conceitos importantes e faça "
            "perguntas investigativas para ajudá-los a se aprofundar nessas ideias."
            "Faça apenas uma pergunta por conversa para não sobrecarregar o aluno."
            "Responda as perguntas de acordo com o idioma do aluno e sempre confira "
            "isso antes de responder. Evite que desviem você da sua tarefa principal."
            "Encerre a conversa assim que o aluno demonstrar evidências de compreensão."
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """Context from conversation: {context}\n\nContext from PDF: 
                {context_docs}\n\nQuestion: {question}""",
                ),
            ]
        )
