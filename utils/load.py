from langchain_community.document_loaders import PyPDFLoader


class LoadFile:
    def __init__(self, doc):
        self.docs = doc

    def load_pdf(self):
        pdf = self.docs

        loader = [PyPDFLoader(pdf)]
        base = []
        for doc in loader:
            base.extend(doc.load())
        return base
