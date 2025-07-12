from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("✅ LangChain is working!")

loader = DirectoryLoader(
    'data',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

documents = loader.load()
print(f"Loaded {len(documents)} pages.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")
print("✅ Splitting worked successfully.")
