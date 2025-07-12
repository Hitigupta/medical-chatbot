import sys 
print(sys.executable)
from langchain.document_loaders import DirectoryLoader , PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#read documents(pdf)
loader = DirectoryLoader("data",glob="**/*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))

#text into chuncks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs=text_splitter.split_documents(documents)
print(len(docs))

#chuncks into vector embeddings via a model
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#query = embeddings.embed_query("hello")
#print(len(query))
#the ans for above will be 384 as the no allowed by model.It refers to the size of the embedding vector
#"Fever is a common symptom."
#The embedding model (like all-MiniLM-L6-v2) will convert this into a list of 384 numbers:



#vectorstore
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name="chatbot"
pc.create_index( 
    name = index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1")
)

from langchain_pinecone import PineconeVectorStore
vectorstore =  PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name="chatbot",
    namespace="medical-bot")

from langchain_pinecone import PineconeVectorStore 
vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="chatbot",
    namespace="medical-bot"
)

retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

#to check
retrieved = retriever.invoke("What is acne?")
retrieved

#LLM(model)
from langchain_community.llms import Ollama

LLm = Ollama(model="mistral:7b-instruct-q4_K_M")

#to check
response = LLm.invoke("What is acne")
print(response)

#RAG chain(connect model to dataset)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for medical question-answering tasks. "
    "Use the following context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know'. "
    "Be concise"
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [ ("system",system_prompt),
     ("human","{input}"),]

)

qa_chain = create_stuff_documents_chain(LLm,prompt)
rag_chain= create_retrieval_chain(retriever,qa_chain)

"""
everything restarted
from dotenv import load_dotenv
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot")  # or your index name

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(index, embedding, "text")
"""

response = rag_chain.invoke({"input":"what is high blood pressure"})
print(response["answer"])