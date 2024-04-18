import random 
# random.seed(42)

from typing import Any

from src.mmr import mmr
from src.embedder import Embedder
from src.text_cleaner import TextCleaner
from src.resume_reader import MyPDFReader
from src.extractor import KeyphraseExtractionPipeline

from src.extractor import StatisiticalExtractor
from src.utils import FilterSimilar


class KeyWordDiversifyer:
    def __init__(self, pdf_path: str, top_n: float = 0.0) -> None:
        """Generate keywords using machine learning and statistical approrach and 
           diversify the keywrods using MMR.

        Args:
            pdf_path (str): PATH to the pdf file.
            top_n (float, optional): Number of keywords that you want. Defaults to 0.0.

        Returns: 
            list(list): Keywords and similarity metric.
        """
        self.pdf_path = pdf_path
        try:
            # read the pdf file
            temp = MyPDFReader()(self.pdf_path)
            # get the text and page number
            self.text = temp["text"]
            # get the page number
            self.page_number = temp["page_number"]

        except Exception as e:
            # raise the exception
            raise Exception(f"Cannot Read the PDF, something happened : {e}")

        # initialize the keyphrase extraction pipeline
        self.extractor = KeyphraseExtractionPipeline(
            model="ml6team/keyphrase-extraction-distilbert-inspec"
        )
        # total number of keywords
        self.top_n = top_n

        # initialize the text cleanser
        self.cleaner = TextCleaner()

        # initialize the statistical extractor
        self.stats_extractor = StatisiticalExtractor()

        # initialize the keywords filterer
        self.filter = FilterSimilar(threshold=0.9)

    def __call__(self, diversity) -> Any:
        # try:
        self.text = self.cleaner(self.text)

        # get the staistical keywords 
        extractive_keywords = list(self.stats_extractor(text=self.text))

        # get the abstractive keywords 
        abstractive_keywords = list(self.extractor(self.text))

        # temp combined 
        temp = extractive_keywords + abstractive_keywords

        # shuffle the random list 
        random.shuffle(temp)
        
        len_half = int(len(temp) / 2)

        # filter similar keywords from both 
        keywords = self.filter(keywords_1=temp[0:len_half] , keywords_2=temp[len_half:-1])
  
        if diversity >= 0.1:
            # keywords embedder class
            embedder = Embedder()

            # key embedding for each of the keywords
            keywords_embedding = embedder(keywords)

            # get the document embedding
            document_embedding = embedder(self.text)

            # remove the embedder obejct
            del embedder

            # call the MMR function
            keywords_mmr = mmr(
                doc_embedding=document_embedding,
                word_embeddings=keywords_embedding,
                words=keywords,
                top_n=self.top_n,
                diversity=diversity,
            )

            return keywords_mmr

        else:
            # if no diversity, return the normal keywords
            return [(x, 0.0) for x in keywords[0 : self.top_n]]

