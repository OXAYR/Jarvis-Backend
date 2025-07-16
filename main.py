from fastapi import FastAPI, UploadFile, File, Form

# Core LangChain functionality
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory

# LangChain community integrations (LLM, Embeddings, Document Loaders, Vector Store)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

import os


app = FastAPI()
llm = Ollama(model="llama3")
memory = ConversationBufferMemory()
chat_chain = ConversationChain(llm=llm, memory=memory, verbose=True)

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    response = chat_chain.run(prompt)
    return {"response": response}

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...), question: str = Form(...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file.filename)
    docs = loader.load_and_split()
    embeddings = OllamaEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    answer = rag_chain.run(question)
    os.remove(file.filename)
    return {"response": answer}