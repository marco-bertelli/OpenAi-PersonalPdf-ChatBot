from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Runelab Chatbot", page_icon=":robot_face:", layout="wide")
    st.header("PDF Runelab Chatbot")

    pdf  = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # divisione in chunk
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=400,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Chiedimi qualcosa sul tuo pdf?"):
        
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with get_openai_callback() as cb:
                    docs = knowledge_base.similarity_search(prompt, k=1)
                    response = chain.run(input_documents=docs, question=prompt)
                
                    print(cb)
                    
                    response = f"Bot: {response}"
            
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == '__main__':
    main()