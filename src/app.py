import streamlit as st
from pathlib import Path
from document_store import DocumentStore
from rag_system import EnhancedRAG
import PyPDF2
import os
from dotenv import load_dotenv

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="RAG-SEK",
    page_icon="🎓",  # You can use an emoji or path to your favicon
    layout="wide"
)

# Load environment variables
load_dotenv()

def initialize_rag():
    if 'rag' not in st.session_state:
        document_store = DocumentStore()
        rag = EnhancedRAG(
            api_key=os.getenv("OPENAI_API_KEY"),
            document_store=document_store
        )
        
        pdfs = document_store.get_all_pdfs()
        if pdfs:
            st.warning("Processing documents... This might take a minute.")
            progress_bar = st.progress(0)
            
            for i, pdf_path in enumerate(pdfs):
                rag.process_pdf(pdf_path)
                progress_bar.progress((i + 1) / len(pdfs))
            
            progress_bar.empty()
            st.success("Processing complete!")
            
        st.session_state.rag = rag

def main():
    st.title("Software Engineering Q&A System University of Kelaniya")
    
    initialize_rag()
    
    # Sidebar for document management
    with st.sidebar:
        st.image("assets/logo.png", width=200)  # Adjust width as needed
        st.header("Document Management")
        
        # Upload new documents
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                file_path = st.session_state.rag.document_store.save_uploaded_pdf(
                    file.read(),
                    file.name
                )
                st.session_state.rag.process_pdf(file_path)
                st.success(f"Processed: {file.name}")
        
        # Show loaded documents with remove buttons
        st.subheader("Loaded Documents")
        for pdf in st.session_state.rag.document_store.get_all_pdfs():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(pdf.name)
            with col2:
                if st.button("🗑️", key=f"remove_{pdf.name}", help="Remove this document"):
                    # First remove from RAG system
                    st.session_state.rag.remove_document(pdf.name)
                    # Then remove the file
                    if st.session_state.rag.document_store.remove_pdf(pdf.name):
                        st.session_state.pop('messages', None)
                        st.rerun()
    
    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"📄 {source.metadata.file_name} - Page {source.metadata.page_number}")
                        st.write(f"Relevance: {source.relevance_score:.2f}")
                        st.text(source.content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response and sources
        response, sources = st.session_state.rag.chat(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
            with st.expander("Sources"):
                for source in sources:
                    st.write(f"📄 {source.metadata.file_name} - Page {source.metadata.page_number}")
                    st.write(f"Relevance: {source.relevance_score:.2f}")
                    st.text(source.content)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()
