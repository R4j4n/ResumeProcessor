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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('stopwords')
# nltk.download('punkt')


# Local import
from embedder import Embedder


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

        
class StatisticalExtractor:

    def __call__(self, text) -> list[str]:
        words_tokenized = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        words = [word for word in words_tokenized if word.lower() not in stop_words]

        freq_dist = nltk.FreqDist(words)
        
        sorted_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

        # usually, 1-5% of the words in a text can be considered as keywords. Therefore, We'll consider 5% of the total word count
        N = int(round(0.05*len(words)))+1

        words_with_frequency = sorted_words[:N] #list[tuple[Any, int]]
        frequent_words = [el[0] for el in words_with_frequency]

        """It isn't ideal to use the term frequency in order to identify the keywords, So, using rake"""
        rake_object = Rake()

        rake_object.extract_keywords_from_text(text)

        rake_keywords = rake_object.get_ranked_phrases_with_scores()

        
        rake_keywords_with_scores = rake_keywords[:N] #List[Tuple[float, Sentence]]
        rake_keywords = [el[-1] for el in rake_keywords_with_scores]

        return self.statistical_selector(frequent_words, rake_keywords)
    
    def statistical_selector(self, keywords1, keywords2) -> list[str]:
        final_keywords = keywords2
        for word1 in keywords1:
            found = False
            for phrase in final_keywords:
                if word1 in phrase:
                    found = True
            if found == False:
                final_keywords.append(word1)

        return final_keywords
    

class KeywordsAggregator:
    def __call__(self, *lists):
        # Combine all the lists into one
        phrases = [phrase for lst in lists for phrase in lst]

        # # Vectorize the phrases
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(phrases)
        vectorizer = Embedder()
        vectors = vectorizer(phrases)

        # Calculate the similarity matrix
        similarity_matrix = cosine_similarity(vectors)

        # Cluster the phrases
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7)
        clustering_model.fit(similarity_matrix)

        # Get the cluster labels for each phrase
        labels = clustering_model.labels_

        # Create a dictionary where the keys are the cluster labels and the values are lists of phrases in that cluster
        clusters = {i: [] for i in range(clustering_model.n_clusters_)}
        for label, phrase in zip(labels, phrases):
            clusters[label].append(phrase)

        # Choose a representative for each cluster
        representatives = [min(cluster, key=len) for cluster in clusters.values()]

        return representatives