import re
from pathlib import Path
import ast
from pathlib import Path
from functools import wraps
from langchain_community.document_loaders import PyPDFLoader


class MyPDFReader:

    def __call__(self, file_pth) -> str:

        # Setp 1: Check if file exists
        pdf_path = Path(file_pth)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Step 2: Read the PDF
        loader = PyPDFLoader(str(pdf_path))
        
        pages = loader.load_and_split()
        # Create a new list for the modified content
        modified_documents = []
        for i, doc in enumerate(pages, start=1):
            modified_documents.append(doc.page_content)

        text = "\n".join(modified_documents)

        if len(pages) >= 1 and len(text) > 50:

            return {
                "page_number" : len(pages), 
                "text" : "\n".join(modified_documents)
            }
        
        else:
            raise Exception("No text found in the PDF")
        
