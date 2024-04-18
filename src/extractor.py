import ssl
import yake
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from transformers.pipelines import AggregationStrategy
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



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


class StatisiticalExtractor:
    def __init__(
        self,
        language="en",
        max_ngram_size=3,
        deduplication_threshold=0.9,
        deduplication_algo="seqm",
        windowSize=1,
        numOfKeywords=20,
    ) -> None:
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.deduplication_algo = deduplication_algo
        self.windowSize = windowSize
        self.numOfKeywords = numOfKeywords

        self.extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            dedupFunc=self.deduplication_algo,
            windowsSize=self.windowSize,
            top=self.numOfKeywords,
            features=None,
        )

    def __call__(self, text: str) -> list:
        keywords =  self.extractor.extract_keywords(text)
        return [x[0] for x in keywords]



