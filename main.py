from fastapi import FastAPI, UploadFile, File, Form
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
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