from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.load import LoadFile
from src.language_model.llm import LoadLLM

class RetrievalAugmentedGeneration:
    def __init__(self, api_key: str, pdf, temperature, top_k, top_p):
        self.key = api_key
        self.pdf = pdf
        self.load_model = LoadLLM(
            model_name="learnlm-2.0-flash-experimental",
            api_key=api_key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        self.model = self.load_model.get_llm()
        self.prompt = self.load_model.prompt()
        self._retriever_instance = None
        self._vector_store = None
        
    def prepare_docs(self):
        docs = LoadFile(self.pdf).load_pdf()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        splitted_docs = text_splitter.split_documents(docs)
        return splitted_docs
        
    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.key
        )
        
    def vector_store(self):
        if self._vector_store is None:
            docs = self.prepare_docs()
            embeddings = self.get_embeddings()
            self._vector_store = FAISS.from_documents(
                documents=docs,
                embedding=embeddings
            )
        return self._vector_store
        
    def retriever(self):
        if self._retriever_instance is None:
            vector = self.vector_store()
            self._retriever_instance = vector.as_retriever()
        return self._retriever_instance
        
    def chain(self):
        def get_docs(query):
            retriever = self.retriever()
            return retriever.get_relevant_documents(query)
        
        rag_chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
                "context_docs": lambda x: get_docs(x["question"])
            }
            | create_stuff_documents_chain(
                llm=self.model,
                prompt=self.prompt,
                document_variable_name="context_docs"
            )
        )
        
        return rag_chain