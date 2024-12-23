import streamlit as st
import os
import tempfile
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI

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

def process_documents(uploaded_files):
    """Process uploaded documents and create vector database"""
    # Create temporary directory for uploaded files
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files to temporary directory
    for file in uploaded_files:
        file_path = os.path.join(st.session_state.temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
    # Load and process documents
    loader = DirectoryLoader(st.session_state.temp_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create vector database
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
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    
    # Setup language model
    turbo_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo',
        openai_api_key=st.session_state.api_key
    )
    
    # Create QA chain
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
    
    # API Key input
    api_key = st.text_input("Enter OpenAI API Key:", type="password", key="api_key_input")
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text files", 
        type=['txt'], 
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
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Main content area
st.title("Document Q&A System")

if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif not st.session_state.has_documents:
    st.warning("Please upload and process some documents in the sidebar.")
else:
    # Question input and response
    question = st.text_input("Ask a question about your documents:", key="question_input")
    
    if question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain(question)
            
            # Display answer
            st.header("Answer:")
            st.write(response['result'])
            
            # Display sources
            st.header("Sources:")
            for source in response["source_documents"]:
                st.write(f"- {source.metadata['source']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Cleanup on session end
def cleanup():
    """Clean up temporary directories"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    if st.session_state.db_dir and os.path.exists(st.session_state.db_dir):
        shutil.rmtree(st.session_state.db_dir)

# Register cleanup
import atexit
atexit.register(cleanup)