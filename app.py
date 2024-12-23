import streamlit as st
import os
import tempfile
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from PyPDF2 import PdfReader
import pandas as pd

# Configure Streamlit page
st.set_page_config(page_title="Document Q&A", layout="wide")

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'db_dir' not in st.session_state:
    st.session_state.db_dir = tempfile.mkdtemp()
if 'has_documents' not in st.session_state:
    st.session_state.has_documents = False

# File loader mapping
LOADER_MAPPING = {
    'txt': TextLoader,
    'pdf': PyPDFLoader,
    'docx': Docx2txtLoader,
    'doc': Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = list(LOADER_MAPPING.keys())

def get_file_extension(file_name):
    """Get the file extension from a filename."""
    return file_name.split('.')[-1].lower()

def load_document(file_path):
    """Load a document using the appropriate loader based on file extension."""
    ext = get_file_extension(file_path)
    if ext not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    loader_class = LOADER_MAPPING[ext]
    
    try:
        loader = loader_class(file_path)
        docs = loader.load()
        
        # Add page numbers for PDFs
        if ext == 'pdf':
            pdf_reader = PdfReader(file_path)
            total_pages = len(pdf_reader.pages)
            for i, doc in enumerate(docs):
                doc.metadata['page_number'] = i + 1
                doc.metadata['total_pages'] = total_pages
                doc.metadata['file_name'] = os.path.basename(file_path)
        
        # For non-PDF documents, store filename and approximate location
        filename = os.path.basename(file_path)
        for i, doc in enumerate(docs):
            doc.metadata['file_name'] = filename
            # For text files, estimate position
            if ext == 'txt':
                doc.metadata['position'] = f"Section {i + 1}"
            
        return docs
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []

def process_documents(uploaded_files):
    """Process uploaded documents and create vector database"""
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.temp_dir = tempfile.mkdtemp()
    
    all_documents = []
    
    for file in uploaded_files:
        try:
            file_path = os.path.join(st.session_state.temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            documents = load_document(file_path)
            all_documents.extend(documents)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    if not all_documents:
        raise ValueError("No documents were successfully processed")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    embedding = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=st.session_state.db_dir
    )
    vectordb.persist()
    
    return vectordb

def setup_qa_chain(vectordb):
    """Setup the QA chain with the vector database"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    
    turbo_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo',
        openai_api_key=st.session_state.api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=turbo_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

# Sidebar for API key and file upload
with st.sidebar:
    st.title("Setup")
    
    api_key = st.text_input("Enter OpenAI API Key:", type="password", key="api_key_input")
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.subheader("Upload Documents")
    st.write(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    uploaded_files = st.file_uploader(
        "Upload your documents", 
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files and st.session_state.api_key:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    vectordb = process_documents(uploaded_files)
                    st.session_state.qa_chain = setup_qa_chain(vectordb)
                    st.session_state.has_documents = True
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Main content area
st.title("Document Q&A System")

if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif not st.session_state.has_documents:
    st.warning("Please upload and process some documents in the sidebar.")
else:
    question = st.text_input("Ask a question about your documents:", key="question_input")
    
    if question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain(question)
            
            st.header("Answer:")
            st.write(response['result'])
            
            st.header("Sources:")
            for source in response["source_documents"]:
                file_name = source.metadata.get('file_name', os.path.basename(source.metadata['source']))
                if 'page_number' in source.metadata:
                    st.write(f"- {file_name} (Page {source.metadata['page_number']} of {source.metadata['total_pages']})")
                elif 'position' in source.metadata:
                    st.write(f"- {file_name} ({source.metadata['position']})")
                else:
                    st.write(f"- {file_name}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def cleanup():
    """Clean up temporary directories"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    if st.session_state.db_dir and os.path.exists(st.session_state.db_dir):
        shutil.rmtree(st.session_state.db_dir)

import atexit
atexit.register(cleanup)