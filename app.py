from flask import Flask, request, render_template
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

system_prompt = (
    "You are an assistant for medical question-answering tasks. "
    "Use the following context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know'. "
    "Be concise"
    "\n\n{context}"
)



# Load the embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="chatbot",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

# Set up the LLM (you can change model name if using something else)
LLm = Ollama(model="mistral")
prompt = ChatPromptTemplate.from_messages(
    [ ("system",system_prompt),
     ("human","{input}"),]

)

# Create the Retrieval Chain
qa_chain = create_stuff_documents_chain(LLm,prompt)
rag_chain= create_retrieval_chain(retriever,qa_chain)

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        result = rag_chain.invoke({"input":user_input})
        response = result["answer"]
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
