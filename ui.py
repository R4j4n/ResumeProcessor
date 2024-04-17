import os
import streamlit as st
from src.viz import create_star_graph

if 'save_pth' not in st.session_state:
    st.session_state.save_pth = None

if 'key_words' not in st.session_state:
    st.session_state.key_words = []

if 'process' not in st.session_state:
    st.session_state.process = False



# Set page configuration
st.set_page_config(layout="wide")

# Title of the page
st.title("Resume Keywords Analyzer.")

# Import the PDF viewer from the library
from streamlit_pdf_viewer import pdf_viewer



# Assuming the main.py has necessary class definitions
from main import KeyWordDiversifyer

# Create two columns for the layout
container_pdf, container_chat = st.columns([40, 60])

# Path where uploaded PDFs will be stored
uploads_dir = "uploads"
os.makedirs(uploads_dir, exist_ok=True)

with container_pdf:

    pdf_file = st.file_uploader("Upload PDF file", type='pdf')

    if pdf_file:
        # Save the PDF to the uploads directory
        file_path = os.path.join(uploads_dir, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getvalue())

        # Display the PDF file
        pdf_viewer(input=pdf_file.getvalue(), width=700)

        # Define a variable for the save path (optional, for further use)
        st.session_state.save_pth = file_path

        if st.session_state.save_pth:
            diversity = st.slider(label="Select how much diversity you want in keywords: ",min_value=0.0,max_value=1.0, step=0.1)

            if st.button("Process"):
                st.session_state.process = True

    
with container_chat:
    if st.session_state.process:
        with st.spinner("Generating ........."):
            k_words = KeyWordDiversifyer(pdf_path=st.session_state.save_pth, top_n=10)
            st.session_state.key_words = k_words(diversity=diversity)

        st.pyplot(fig=(create_star_graph(keyword_similarity_data=st.session_state.key_words)))