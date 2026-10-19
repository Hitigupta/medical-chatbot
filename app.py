import os

from flask import Flask, request, render_template
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

rag_chain = None

system_prompt = (
    "You are an assistant for medical question-answering tasks. "
    "Use the following context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know'. "
    "Be concise."
    "\n\n{context}"
)


def initialize_rag():
    global rag_chain

    if not PINECONE_API_KEY:
        app.logger.warning("PINECONE_API_KEY is not set. RAG is disabled.")
        return None

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _ = pc  # Keep the client alive for the process lifecycle.

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="chatbot",
            embedding=embeddings,
            namespace="",
        )
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        docs = retriever.invoke("What is fever?")
        print("RETRIEVED DOCUMENTS:", len(docs))

        for i, doc in enumerate(docs):
            print(f"\n--- Document {i + 1} ---")
            print(doc.page_content[:500])

        llm = ChatOpenAI(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        app.logger.info("RAG chain initialized successfully.")
        return rag_chain
    except Exception as exc:
        app.logger.exception("RAG initialization failed: %s", exc)
        return None


initialize_rag()


@app.route("/", methods=["GET", "POST"])
def index():
    response = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()

        if not user_input:
            response = "Please enter a question or symptoms to check."
        elif rag_chain is None:
            response = (
                "The medical assistant is not configured. "
                "Check the server configuration and logs."
            )
        else:
            try:
                result = rag_chain.invoke({"input": user_input})
                response = result.get("answer", "I don't know.")
            except Exception as exc:
                app.logger.exception("Query failed: %s", exc)
                response = "I could not answer that question right now. Please try again later."

    return render_template("index.html", response=response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
