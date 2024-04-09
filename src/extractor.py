from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake

from transformers.pipelines import AggregationStrategy
import numpy as np


import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')



# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])



class StatisticalSelector:
    def __call__(self, keywords1, keywords2) -> list[str]:
        final_keywords = keywords2
        for word1 in keywords1:
            found = False
            for phrase in final_keywords:
                if word1 in phrase:
                    found = True
            if found == False:
                final_keywords.append(word1)

        return final_keywords


class StatisticalExtractor:

    def __call__(self, text) -> list[str]:
        words_tokenized = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        words = [word for word in words_tokenized if word.lower() not in stop_words]

        freq_dist = nltk.FreqDist(words)
        
        sorted_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

        # usually, 1-5% of the words in a text can be considered as keywords. Therefore, We'll consider 4% of the total word count
        N = int(round(0.04*len(words)))+1

        words_with_frequency = sorted_words[:N] #list[tuple[Any, int]]
        frequent_words = [el[0] for el in words_with_frequency]

        """It isn't ideal to use the term frequency in order to identify the keywords, So, using rake"""
        rake_object = Rake()

        rake_object.extract_keywords_from_text(text)

        rake_keywords = rake_object.get_ranked_phrases_with_scores()

        
        rake_keywords_with_scores = rake_keywords[:N] #List[Tuple[float, Sentence]]
        rake_keywords = [el[-1] for el in rake_keywords_with_scores]

        return StatisticalSelector()(frequent_words, rake_keywords)