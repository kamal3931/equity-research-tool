import pickle
import os
import streamlit as st
import dill
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain_openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import time
import faiss
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
st.title("Equity reseach tool")
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faissindex"

main_placeholder = st.empty()
st.button("submit")
st.sidebar.image(image="img.png",width=285)
query = main_placeholder.text_input("Question: ")


llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    Embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, Embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    vectorstore_openai.save_local(file_path)

if query:
    if os.path.exists(file_path):
        faissindex = FAISS.load_local(file_path,embeddings = OpenAIEmbeddings(),allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faissindex.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

# st.write("the below answer is based on the articles you mentioned\n:",X)

