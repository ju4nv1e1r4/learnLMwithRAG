from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.language_model.llm import LoadLLM


class DocumentManager:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_pdf(self):
        loader = [PyPDFLoader(self.pdf_path)]
        base = []
        for doc in loader:
            base.extend(doc.load())
        return base

    def prepare_docs(self):
        docs = self.load_pdf()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        splitted_docs = text_splitter.split_documents(docs)
        return splitted_docs


class EmbeddingManager:
    def __init__(self, api_key: str):
        self.key = api_key
        self._vector_store = None

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=self.key
        )

    def create_vector_store(self, documents):
        embeddings = self.get_embeddings()
        self._vector_store = FAISS.from_documents(
            documents=documents, embedding=embeddings
        )
        return self._vector_store

    def get_retriever(self, vector_store=None):
        if vector_store:
            return vector_store.as_retriever()
        elif self._vector_store:
            return self._vector_store.as_retriever()
        else:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store first."
            )


class RetrievalAugmentedGeneration:
    def __init__(self, api_key: str, pdf_path, temperature=0.7, top_k=0.0, top_p=0.0):
        self.key = api_key
        self.pdf_path = pdf_path
        self.load_model = LoadLLM(
            model_name="learnlm-2.0-flash-experimental",
            api_key=api_key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        self.model = self.load_model.get_llm()
        self.prompt = self.load_model.prompt()
        self._retriever_instance = None

        self.doc_manager = DocumentManager(pdf_path)
        self.embedding_manager = EmbeddingManager(api_key)

    def prepare_docs(self):
        return self.doc_manager.prepare_docs()

    def retriever(self):
        if self._retriever_instance is None:
            docs = self.prepare_docs()
            vector_store = self.embedding_manager.create_vector_store(docs)
            self._retriever_instance = self.embedding_manager.get_retriever(
                vector_store
            )
        return self._retriever_instance

    def chain(self):
        def get_docs(query):
            retriever = self.retriever()
            return retriever.get_relevant_documents(query)

        rag_chain = {
            "question": lambda x: x["question"],
            "context": lambda x: x["context"],
            "context_docs": lambda x: get_docs(x["question"]),
        } | create_stuff_documents_chain(
            llm=self.model, prompt=self.prompt, document_variable_name="context_docs"
        )

        return rag_chain
