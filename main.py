from typing import Any
from src.mmr import mmr
from src.embedder import Embedder
from src.text_cleaner import TextCleaner
from src.resume_reader import MyPDFReader
from src.extractor import KeyphraseExtractionPipeline, StatisticalExtractor
from src.extractor import KeywordsAggregator


class KeyWordDiversifyer:
    def __init__(self, pdf_path: str, top_n: float = 0.0) -> None:
        self.pdf_path = pdf_path
        try:
            temp = MyPDFReader()(self.pdf_path)
            self.text = temp["text"]
            self.page_number = temp["page_number"]

        except Exception as e:
            raise Exception(f"Cannot Read the PDF, something happened : {e}")

        self.extractor = KeyphraseExtractionPipeline(
            model="ml6team/keyphrase-extraction-distilbert-inspec"
        )
        self.statisitcal_extractor = StatisticalExtractor()

        self.cleaner = TextCleaner()

        self.top_n = top_n

    def __call__(self, diversity) -> Any:

        # try:
        # self.text = self.cleaner(self.text)

        keywords = KeywordsAggregator()(list(self.extractor(self.text)) , list(self.statisitcal_extractor(text=self.text)))    

        if diversity >= 0.1:
            # generate keywords and words embedding
            embedder = Embedder()

            keywords_embedding = embedder(keywords)
            document_embedding = embedder(self.text)

            del embedder

            keywords_mmr = mmr(
                doc_embedding=document_embedding,
                word_embeddings=keywords_embedding,
                words=keywords,
                top_n=self.top_n,
                diversity=diversity,
            )

            return keywords_mmr

        else:
            return [(x, 0.0) for x in keywords[0 : self.top_n]]
        # except:
        #     raise Exception("Cannot Extract Keyphrases")


if __name__ == "__main__":
    kd = KeyWordDiversifyer("samples/V4_Rajan_T.pdf", top_n=10)
    kwds = kd(diversity=0.6)
    print(kwds)
