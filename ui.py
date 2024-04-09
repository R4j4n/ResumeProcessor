import streamlit as st
st.set_page_config(layout="wide")
from streamlit_pdf_viewer import pdf_viewer
st.title("Resume Keywords Analyzer.")



container_pdf, container_chat = st.columns([40, 60])


with container_pdf:
    pdf_file = st.file_uploader("Upload PDF file", type=('pdf'))

    if pdf_file:
        binary_data = pdf_file.getvalue()
        pdf_viewer(input=binary_data,
                   width=700)