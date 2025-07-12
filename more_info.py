from langchain.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader("new data",glob="**/*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))
#read documents(pdf)

#text into chuncks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs=text_splitter.split_documents(documents)
print(len(docs))

#chuncks into vector embeddings via a model
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Documents loaded and vector store saved!")

import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

if "chatbot" not in pc.list_indexes().names():
    pc.create_index(
        name="chatbot",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created.")

    # Create new index with docs
    from langchain_pinecone import PineconeVectorStore 
    vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="chatbot",
    namespace="medical-bot")
    vectorstore.add_documents(docs)
    print("✅ Documents uploaded to new index.")
else:
    # Connect to existing index and add new docs
    vectorstore = PineconeVectorStore.from_existing_index(index_name="chatbot", embedding=embeddings)
    vectorstore.add_documents(docs)
    print("✅ New documents added to existing index.")