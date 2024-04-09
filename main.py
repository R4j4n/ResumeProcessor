from src.resume_reader import MyPDFReader

class KeyWordDiversifyer:
    def __init__(self, pdf_path : str) -> None:
        self.pdf_path = pdf_path
        try: 
            temp = MyPDFReader()(self.pdf_path)
            self.text = temp["text"]
            self.page_number = temp["page_number"]

        except Exception as e:
            raise Exception(f"Cannot Read the PDF, something happened : {e}")
        

