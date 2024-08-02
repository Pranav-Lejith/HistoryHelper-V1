import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Pre-loaded PDFs (these should be paths to your PDF files)
pdf_files = {
    "Chapter 1": "./chapters/chapter1.pdf",
    "Chapter 2": "./chapters/chapter2.pdf",
    "Chapter 3": "./chapters/chapter3.pdf",
    "Chapter 4": "./chapters/chapter4.pdf",
    "Chapter 5": "./chapters/chapter5.pdf"
}

# To maintain chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    You are a historian with expertise in answering questions related to history. Answer the question as detailed as possible 
    from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, 
    "The answer is not available in the context," and do not provide incorrect information.

    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def display_chat():
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://img.icons8.com/color/48/000000/user.png" width="30" style="margin-right: 10px;">
                    <div>
                        User: {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <img src="https://img.icons8.com/fluency/48/000000/chatbot.png" width="30" style="margin-right: 10px;">
                    <div>
                        Assistant: {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="PDF Reader", page_icon=":book:")

    st.header("PDF Reader")

    with st.sidebar:
        st.title("Menu:")
        api_key = st.text_input("Enter your Google API Key", type="password")
        chapter_choice = st.selectbox("Choose a Chapter", options=list(pdf_files.keys()))

        if st.button("Process Chapter", key="process_chapter"):
            if api_key:
                with st.spinner("Processing..."):
                    pdf_path = pdf_files[chapter_choice]
                    raw_text = get_pdf_text(pdf_path)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")
            else:
                st.error("Please enter your Google API Key")

        st.info("This chatbot uses Google Generative AI model for conversational responses.")
        st.info("Created By Pranav Lejith(Amphibiar)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Submit Question", key="submit_question"):
        if user_question and api_key:
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})

            # Get response from the model
            response = user_input(user_question, api_key)
            
            # Add model response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display chat history
            display_chat()
        elif not api_key:
            st.error("Please enter your Google API Key")

if __name__ == "__main__":
    main()
