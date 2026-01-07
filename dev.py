import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# ---------------- Setup ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG ChatBot for Prop Data", layout="wide")
st.title("RAG ChatBot for Prop Data")

st.write("PDF exists:", os.path.exists("./Medical_book.pdf"))
st.write("GROQ key present:", bool(GROQ_API_KEY))

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ---------------- Vector store ----------------
@st.cache_resource(show_spinner=True)
def get_vectorstore():
    pdf_path = "./Medical_book.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb

# ---------------- RAG chain ----------------
@st.cache_resource(show_spinner=False)
def get_rag_chain():
    st.write("Building RAG chain...")
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers using ONLY the provided context. "
         "If the answer is not contained in the context, say you do not know."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])

    def rag_answer(question: str) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    return rag_answer

rag_answer = get_rag_chain()

# ---------------- Chat loop ----------------
prompt = st.chat_input("Ask a question about Prop data")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_answer(prompt)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {str(e)}")
